from __future__ import annotations

import importlib.util
import math
import re
import shutil
import struct
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

from native_runtime.compact_summary_digests import (
    BEHAVIOR_ROW_FIELDS,
    LINEAGE_GENOME_FIELDS,
    LINEAGE_RECORD_FIELDS,
    LINEAGE_STATS_FIELDS,
    behavior_rows_digest_sha256,
    lineage_records_digest_sha256,
)
from native_runtime.compact_summary_reporting_projection import (
    REPORTING_RETAINED_LAYOUT_VERSION,
)
from native_runtime.public_metrics_overlay import _native_lineage_record
from perf.cuda_architecture_b_reporting import _base_fitness


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    ROOT
    / "engine"
    / "perf"
    / "cuda_architecture_b_compact_summary_reporting_authority_v1.cuh"
)
CUDA_SOURCE = HEADER.with_suffix(".cu")
BINDING = (
    ROOT
    / "tests"
    / "perf"
    / "test_eiel_cuda_architecture_b_compact_summary_reporting_authority_v1_binding.cpp"
)
COMPILE_TEST = (
    ROOT
    / "tests"
    / "perf"
    / "test_eiel_cuda_architecture_b_compact_summary_reporting_authority_v1_compile.cu"
)
PROBE = (
    ROOT
    / "tools"
    / "scripts"
    / "eiel_cuda_architecture_b_compact_summary_reporting_authority_v1_probe.py"
)


def _load_probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "eiel_reporting_authority_v1_probe_test", PROBE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _f32(value: float) -> float:
    return float(np.float32(value))


def _behavior(
    *,
    agent_id: int,
    cohort: str,
    actions: tuple[int, ...],
    survival: int,
    energy: float,
    offspring: int,
    attacks: int,
    early_survival: int,
    early_energy: float,
    early_injury: float,
    early_weight: float,
    alive: bool,
    kept: bool,
    injury_start: float,
    injury_end: float,
    aggression: float,
) -> dict[str, Any]:
    return {
        "generation": 0,
        "agent_id": agent_id,
        "cohort": cohort,
        "move_n": actions[0],
        "move_s": actions[1],
        "move_e": actions[2],
        "move_w": actions[3],
        "gather": actions[4],
        "rest": actions[5],
        "attack": actions[6],
        "lifespan_steps": survival,
        "energy_collected": float(energy),
        "offspring": offspring,
        "attacks_made": attacks,
        "fitness": _base_fitness(
            survival=survival,
            energy=energy,
            offspring=offspring,
            attacks=attacks,
            early_survival=early_survival,
            early_energy=early_energy,
            early_injury=early_injury,
            early_weight=early_weight,
            alive_end=alive,
        ),
        "alive_end": int(alive),
        "eligible_for_selection": int(cohort == "start" and alive),
        "kept_next": int(kept),
        "injury_end": _f32(injury_end),
        "injury_delta": _f32(injury_end) - _f32(injury_start),
        "aggression_bias": round(_f32(aggression), 8),
    }


def _expected_normal_behavior() -> list[dict[str, Any]]:
    return [
        _behavior(
            agent_id=10,
            cohort="start",
            actions=(2, 0, 0, 0, 0, 0, 0),
            survival=3,
            energy=0.75,
            offspring=0,
            attacks=0,
            early_survival=2,
            early_energy=0.25,
            early_injury=0.0,
            early_weight=1.0,
            alive=False,
            kept=False,
            injury_start=-0.0,
            injury_end=0.5,
            aggression=0.375,
        ),
        _behavior(
            agent_id=11,
            cohort="start",
            actions=(0, 0, 0, 0, 0, 0, 1),
            survival=6,
            energy=2.5,
            offspring=1,
            attacks=1,
            early_survival=3,
            early_energy=1.25,
            early_injury=0.0,
            early_weight=1.0,
            alive=True,
            kept=True,
            injury_start=0.25,
            injury_end=0.125,
            aggression=-0.25,
        ),
        _behavior(
            agent_id=12,
            cohort="born",
            actions=(0, 0, 0, 0, 1, 0, 0),
            survival=5,
            energy=1.5,
            offspring=0,
            attacks=0,
            early_survival=2,
            early_energy=0.25,
            early_injury=0.0,
            early_weight=1.0,
            alive=True,
            kept=False,
            injury_start=0.0,
            injury_end=0.0,
            aggression=round(_f32(0.123456789), 8),
        ),
    ]


def _expected_normal_lineage(
    behavior: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {row["agent_id"]: row for row in behavior}
    return [
        _native_lineage_record(
            agent_id=11,
            founder_id=11,
            lineage_depth=0,
            birth_generation=0,
            birth_tick=0,
            parent_ids=(),
            energy=41.0,
            health=1.0,
            injury=by_id[11]["injury_end"],
            drives=[_f32(1.1), _f32(1.2), _f32(1.3), _f32(1.4)],
            offspring_count=1,
            survival_steps=6.0,
            energy_collected=2.5,
            first_repro_step=4,
            pred_err_count=2,
            pred_err_sum=1.0,
            episode0_pred_err_mean=0.25,
            identity_drift_ema=0.03125,
            drive_jitter=0.0625,
            hidden_size=9,
            memory_dim=5,
            use_gru=True,
            aggression_bias=-0.25,
            exploration_bias=-0.25,
            patch_habitat_lineage_id=8,
            y=3,
            x=4,
        ),
        _native_lineage_record(
            agent_id=12,
            founder_id=12,
            lineage_depth=1,
            birth_generation=0,
            birth_tick=3,
            parent_ids=(),
            energy=40.0,
            health=1.0,
            injury=by_id[12]["injury_end"],
            drives=[_f32(0.1), _f32(0.2), _f32(0.3), _f32(0.4)],
            offspring_count=0,
            survival_steps=5.0,
            energy_collected=1.5,
            first_repro_step=None,
            pred_err_count=1,
            pred_err_sum=0.5,
            episode0_pred_err_mean=0.25,
            identity_drift_ema=0.015625,
            drive_jitter=0.03125,
            hidden_size=8,
            memory_dim=4,
            use_gru=False,
            aggression_bias=round(_f32(0.123456789), 8),
            exploration_bias=-0.125,
            patch_habitat_lineage_id=7,
            y=2,
            x=3,
        ),
    ]


def test_header_freezes_device_neutral_pods_and_exact_budgets() -> None:
    source = HEADER.read_text(encoding="utf-8")
    assert REPORTING_RETAINED_LAYOUT_VERSION in source
    assert "struct alignas(8) GenerationAuthorityHeaderV1" in source
    assert "struct alignas(8) BehaviorAuthorityRowV1" in source
    assert "struct alignas(8) LineageAuthorityRowV1" in source
    assert "sizeof(BehaviorAuthorityRowV1) == 56" in source
    assert "sizeof(LineageAuthorityRowV1) == 136" in source
    assert "retained_bytes_per_generation_cell_v1(kStrictRowCapacityV1) == 12304" in source
    assert "retained_bytes_per_generation_cell_v1(kDynamicRowCapacityV1) == 43024" in source
    for field, offset in _load_probe().LINEAGE_OFFSETS.items():
        assert f"offsetof(LineageAuthorityRowV1, {field}) == {offset}" in source
    assert re.search(r"\b(?:ATen|Torch|torch|malloc|calloc|new)\b", source) is None


def test_header_exposes_all_required_fail_closed_lifecycle_hooks() -> None:
    source = HEADER.read_text(encoding="utf-8")
    for symbol in (
        "initialize_generation_v1",
        "count_action_v1",
        "append_ordinary_birth_v1",
        "seal_death_before_slot_reuse_v1",
        "finalize_pre_turnover_v1",
        "kLedgerOverflowV1",
        "kLedgerDuplicateAgentIdV1",
        "kLedgerSlotAliasV1",
        "kLedgerWriteAfterSealV1",
    ):
        assert symbol in source
    assert "ReportingFitnessConfigV1" in source
    assert "ResidentGenerationSourcesV1" in source
    assert "persistent_turnover_plan" not in source
    assert "reduced_fitness" not in source


def test_nvcc_compiles_framework_free_header_for_sm75(tmp_path: Path) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is unavailable")
    output = tmp_path / "reporting_authority_compile.o"
    completed = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-O3",
            "--fmad=false",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_75,code=sm_75",
            "-c",
            str(COMPILE_TEST),
            "-o",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    assert "ptxas info" in completed.stderr


@pytest.fixture(scope="module")
def qualification(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModuleType, dict, dict]:
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        pytest.skip("CUDA compiler and device are required")
    probe = _load_probe()
    module, manifest = probe.build(tmp_path_factory.mktemp("reporting_authority_v1"))
    receipt = probe.run_probe(module)
    return probe, manifest, receipt


def test_artifact_is_sm75_sass_only_with_resource_receipt(
    qualification: tuple[ModuleType, dict, dict],
) -> None:
    _, manifest, receipt = qualification
    assert manifest["cuda_arch"] == "sm_75"
    assert manifest["fmad"] is False
    assert manifest["retained_layout_version"] == REPORTING_RETAINED_LAYOUT_VERSION
    assert manifest["artifact_identity"] == {
        "sm75_elf_present": True,
        "sass_only": True,
        "ptx_embedded": False,
        "runtime_jit_required": False,
    }
    assert receipt["binary_version"] == 75
    assert receipt["registers_per_thread"] > 0
    assert receipt["local_bytes_per_thread"] >= 0
    assert receipt["shared_bytes_per_block"] == 0
    assert manifest["ptxas"]["spill_store_bytes"] == 0
    assert manifest["ptxas"]["spill_load_bytes"] == 0
    assert receipt["header_bytes"] == 16
    assert receipt["behavior_row_bytes"] == 56
    assert receipt["lineage_row_bytes"] == 136
    assert receipt["strict_bytes_per_generation_cell"] == 12304
    assert receipt["dynamic_bytes_per_generation_cell"] == 43024
    assert receipt["retained_layout_version"] == REPORTING_RETAINED_LAYOUT_VERSION
    assert receipt["lineage_field_offsets"] == _load_probe().LINEAGE_OFFSETS
    assert receipt["abi_layout_verified"] is True


def test_start_birth_death_slot_reuse_matches_accepted_cpu_authorities(
    qualification: tuple[ModuleType, dict, dict],
) -> None:
    probe, _, receipt = qualification
    generation = receipt["fixtures"]["start_born_died_reuse"]["generations"][0]
    assert generation["status"] == 2
    assert generation["behavior_row_count"] == 3
    assert generation["lineage_record_count"] == 2
    actual_behavior = probe.public_behavior_rows(generation)
    expected_behavior = _expected_normal_behavior()
    expected_lineage = _expected_normal_lineage(expected_behavior)
    assert all(set(row) == set(BEHAVIOR_ROW_FIELDS) for row in actual_behavior)
    assert all(set(record) == set(LINEAGE_RECORD_FIELDS) for record in generation["lineage_records"])
    assert all(
        set(record["stats"]) == set(LINEAGE_STATS_FIELDS)
        for record in generation["lineage_records"]
    )
    assert all(
        set(record["genome"]) == set(LINEAGE_GENOME_FIELDS)
        for record in generation["lineage_records"]
    )
    assert actual_behavior == expected_behavior
    assert generation["lineage_records"] == expected_lineage
    expected_behavior_digest = behavior_rows_digest_sha256(expected_behavior)
    expected_lineage_digest = lineage_records_digest_sha256(0, expected_lineage)
    assert behavior_rows_digest_sha256(actual_behavior) == expected_behavior_digest
    assert generation["behavior_rows_digest_sha256"] == expected_behavior_digest
    assert (
        lineage_records_digest_sha256(0, generation["lineage_records"])
        == expected_lineage_digest
    )
    assert generation["lineage_records_digest_sha256"] == expected_lineage_digest
    assert receipt["fixtures"]["start_born_died_reuse"][
        "independent_projection_verified"
    ] is True
    assert [row["_raw_flags"] for row in generation["behavior_rows"]] == [8, 15, 9]


def test_empty_dynamic_overflow_duplicate_and_write_after_seal_fail_closed(
    qualification: tuple[ModuleType, dict, dict],
) -> None:
    _, _, receipt = qualification
    empty = receipt["fixtures"]["empty"]["generations"][0]
    assert (empty["status"], empty["behavior_row_count"], empty["lineage_record_count"]) == (
        2,
        0,
        0,
    )
    overflow = receipt["fixtures"]["overflow"]["generations"][0]
    assert overflow["status"] == 102
    assert overflow["start_id_ceiling"] == 64
    assert overflow["behavior_row_count"] == 64
    assert [row["agent_id"] for row in overflow["behavior_rows"]] == list(range(64))
    duplicate = receipt["fixtures"]["duplicate"]["generations"][0]
    assert duplicate["status"] == 103
    assert duplicate["behavior_row_count"] == 2
    assert [row["agent_id"] for row in duplicate["behavior_rows"]] == [0, 1]
    sealed = receipt["fixtures"]["write_after_seal"]["generations"][0]
    assert sealed["status"] == 109
    assert sealed["behavior_row_count"] == 0
    missing = receipt["fixtures"]["missing_lineage_source"]["generations"][0]
    assert (missing["status"], missing["behavior_row_count"], missing["lineage_record_count"]) == (
        100,
        1,
        0,
    )
    malformed = receipt["fixtures"]["malformed_lineage_source"]["generations"][0]
    assert (
        malformed["status"],
        malformed["behavior_row_count"],
        malformed["lineage_record_count"],
    ) == (112, 1, 0)
    assert receipt["fixtures"]["missing_lineage_source"][
        "independent_projection_verified"
    ] is False
    assert receipt["fixtures"]["malformed_lineage_source"][
        "independent_projection_verified"
    ] is False


def test_identity_drift_zero_is_only_synthesized_when_source_is_unavailable(
    qualification: tuple[ModuleType, dict, dict],
) -> None:
    _, _, receipt = qualification
    normal = receipt["fixtures"]["start_born_died_reuse"]["generations"][0]
    assert [
        record["stats"]["identity_drift_ema"]
        for record in normal["lineage_records"]
    ] == [0.03125, 0.015625]
    unavailable_fixture = receipt["fixtures"]["identity_drift_unavailable"]
    unavailable = unavailable_fixture["generations"][0]
    assert unavailable["status"] == 2
    assert unavailable["lineage_records"][0]["stats"]["identity_drift_ema"] == 0.0
    assert unavailable_fixture["independent_projection_verified"] is True


def test_signed_zero_survives_typed_ledger_until_host_canonicalization(
    qualification: tuple[ModuleType, dict, dict],
) -> None:
    _, _, receipt = qualification
    generation = receipt["fixtures"]["signed_zero"]["generations"][0]
    behavior = generation["behavior_rows"][0]
    lineage = generation["lineage_records"][0]
    assert generation["status"] == 2
    assert math.copysign(1.0, behavior["_injury_start"]) == -1.0
    assert math.copysign(1.0, behavior["injury_end"]) == -1.0
    assert math.copysign(1.0, behavior["energy_collected"]) == -1.0
    assert math.copysign(1.0, lineage["energy"]) == -1.0
    assert math.copysign(1.0, lineage["health"]) == -1.0
    assert math.copysign(1.0, lineage["injury"]) == -1.0
    assert math.copysign(1.0, lineage["stats"]["energy_collected"]) == -1.0
    raw = generation["raw"]
    behavior_offset = 16
    unpacked = struct.Struct("<qddfffiii7BB").unpack_from(raw, behavior_offset)
    assert math.copysign(1.0, unpacked[5]) == -1.0
    lineage_offset = behavior_offset + 64 * 56
    for field, format_code in (
        ("energy", "d"),
        ("energy_collected_exact", "d"),
        ("health", "f"),
        ("injury", "f"),
        ("aggression_bias_f32", "f"),
        ("exploration_bias_f32", "f"),
    ):
        value = struct.unpack_from(
            f"<{format_code}",
            raw,
            lineage_offset + _load_probe().LINEAGE_OFFSETS[field],
        )[0]
        assert math.copysign(1.0, value) == -1.0


def test_host_decoder_rejects_lineage_identity_and_reserved_storage_corruption(
    qualification: tuple[ModuleType, dict, dict],
) -> None:
    probe, _, receipt = qualification
    generation = receipt["fixtures"]["signed_zero"]["generations"][0]
    lineage_offset = 16 + 64 * 56

    identity_corrupt = bytearray(generation["raw"])
    struct.pack_into("<q", identity_corrupt, lineage_offset, 99)
    with pytest.raises(ValueError, match="agent ID disagrees"):
        probe.decode_retained(
            torch.tensor(identity_corrupt, dtype=torch.uint8).reshape(1, -1)
        )

    reserved_corrupt = bytearray(generation["raw"])
    reserved_corrupt[lineage_offset + probe.LINEAGE_OFFSETS["reserved"]] = 1
    with pytest.raises(ValueError, match="reserved storage"):
        probe.decode_retained(
            torch.tensor(reserved_corrupt, dtype=torch.uint8).reshape(1, -1)
        )


def test_g2_rows_are_isolated_and_use_one_final_transfer(
    qualification: tuple[ModuleType, dict, dict],
) -> None:
    _, _, receipt = qualification
    fixture = receipt["fixtures"]["g2_isolation"]
    assert fixture["host_transfers"] == 1
    first, second = fixture["generations"]
    assert first["status"] == second["status"] == 2
    assert first["start_id_ceiling"] == 1
    assert second["start_id_ceiling"] == 11
    assert first["behavior_rows"][0]["agent_id"] == 0
    assert second["behavior_rows"][0]["agent_id"] == 10
    assert first["behavior_rows"][0]["move_n"] == 1
    assert first["behavior_rows"][0]["attack"] == 0
    assert second["behavior_rows"][0]["move_n"] == 0
    assert second["behavior_rows"][0]["attack"] == 2
    assert first["lineage_records"][0]["agent_id"] == 0
    assert second["lineage_records"][0]["agent_id"] == 10
    assert first["behavior_rows_digest_sha256"] != second["behavior_rows_digest_sha256"]
    assert first["lineage_records_digest_sha256"] != second["lineage_records_digest_sha256"]
    assert first["raw"] != second["raw"]
    assert fixture["independent_projection_verified"] is True
    assert receipt["one_host_transfer_per_fixture"] is True
    assert receipt["strict_g2_b1_bytes"] == 24_608
    assert receipt["dynamic_g32_b16_bytes"] == 22_028_288
