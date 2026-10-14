from __future__ import annotations

import importlib.util
import shutil
import struct
import subprocess
import sysconfig
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pybind11
import numpy as np
import pytest

from engine.native_runtime.compact_summary_digests import (
    behavior_rows_digest_sha256,
    lineage_records_digest_sha256,
)
from engine.native_runtime.compact_summary_reporting_projection import (
    project_retained_reporting_authority,
)
from simulation.metrics_pkg.lineage import (
    LineageMetricsState,
    NativeLineageAgentRecord,
    NativeLineageGenomeRecord,
    NativeLineageStatsRecord,
    compute_lineage_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    ROOT
    / "engine"
    / "perf"
    / "native"
    / "cpu_compact_summary_reporting_ledger_v1.hpp"
)
SOURCE = HEADER.with_suffix(".cpp")
HOST_CANONICAL_SOURCE = (
    ROOT
    / "engine"
    / "perf"
    / "cuda_architecture_b_compact_summary_host_canonical_v1.cpp"
)
BINDING = (
    ROOT
    / "tests"
    / "perf"
    / "test_eiel_cpu_compact_summary_reporting_ledger_v1_binding.cpp"
)
MODULE_NAME = "eiel_cpu_compact_summary_reporting_ledger_v1_test"

LINEAGE_CONFIG = {
    "lineage_dominance_threshold": 0.80,
    "lineage_entropy_minimum": 0.10,
    "lineage_reproduction_dominance_threshold": 0.80,
    "lineage_dominance_streak_required": 5,
    "lineage_min_active_lineages": 2,
    "lineage_reproduction_dominance_streak_required": 5,
}


@pytest.fixture(scope="module")
def native_ledger(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 compiler is unavailable")
    build = tmp_path_factory.mktemp("cpu_compact_summary_reporting_ledger_v1")
    extension = build / f"{MODULE_NAME}{sysconfig.get_config_var('EXT_SUFFIX')}"
    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            "-fvisibility=hidden",
            "-Wall",
            "-Wextra",
            f"-I{pybind11.get_include()}",
            f"-I{sysconfig.get_paths()['include']}",
            f"-I{ROOT / 'engine' / 'perf' / 'native'}",
            f"-I{ROOT / 'engine' / 'perf'}",
            str(BINDING),
            str(SOURCE),
            str(HOST_CANONICAL_SOURCE),
            "-o",
            str(extension),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    spec = importlib.util.spec_from_file_location(MODULE_NAME, extension)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native_agent(record: dict[str, Any]) -> NativeLineageAgentRecord:
    stats_input = record["stats"]
    genome_input = record["genome"]
    stats = NativeLineageStatsRecord(
        lineage_id=stats_input["lineage_id"],
        founder_id=stats_input["founder_id"],
        parent_ids=(),
        parent_lineage_ids=(),
        birth_generation=stats_input["birth_generation"],
        birth_tick=stats_input["birth_tick"],
        lineage_depth=stats_input["lineage_depth"],
        offspring_count=stats_input["offspring_count"],
        survival_steps=stats_input["survival_steps"],
        energy_collected=stats_input["energy_collected"],
        first_repro_step=stats_input["first_repro_step"],
        pred_err_count=stats_input["pred_err_count"],
        pred_err_sum=stats_input["pred_err_sum"],
        episode0_pred_err_mean=stats_input["episode0_pred_err_mean"],
        identity_drift_ema=stats_input["identity_drift_ema"],
        drive_jitter=stats_input["drive_jitter"],
        ecological_role="general_agent",
        role_source="default",
        role_confidence=None,
    )
    genome = NativeLineageGenomeRecord(
        hidden_size=genome_input["hidden_size"],
        memory_dim=genome_input["memory_dim"],
        use_gru=genome_input["use_gru"],
        aggression_bias=genome_input["aggression_bias"],
        exploration_bias=genome_input["exploration_bias"],
    )
    return NativeLineageAgentRecord(
        agent_id=record["agent_id"],
        energy=record["energy"],
        health=record["health"],
        injury=record["injury"],
        drives=tuple(record["drives"]),
        genome=genome,
        stats=stats,
    )


def _assert_float_bits_equal(actual: float, expected: float) -> None:
    assert struct.pack("=d", actual) == struct.pack("=d", expected)


def test_framework_free_strict_cxx17_and_fixed_storage(
    native_ledger: ModuleType, tmp_path: Path
) -> None:
    production = HEADER.read_text(encoding="ascii") + SOURCE.read_text(
        encoding="ascii"
    )
    for forbidden in (
        "#include <cuda",
        "cuda_runtime",
        "#include <torch",
        "ATen/",
        "Python.h",
        "pybind11",
        "std::vector",
        "std::unique_ptr",
        "std::shared_ptr",
        "malloc(",
        "calloc(",
        "realloc(",
        "operator new",
    ):
        assert forbidden not in production
    assert "std::array" in production
    assert "initialize_generation_from_slot_state_v1" in production
    assert "initialize_generation_from_active_slot_state_v1" in production
    assert "canonicalize_retained_summary_v1" in production

    receipt = native_ledger.storage_receipt()
    assert receipt == {
        "maximum_row_capacity": 31_104,
        "maximum_slot_capacity": 1024,
        "ledger_bytes": 6_225_320,
        "header_bytes": 16,
        "behavior_row_bytes": 56,
        "lineage_row_bytes": 136,
        "canonical_result_bytes": 360,
    }

    compiler = shutil.which("g++") or shutil.which("clang++")
    assert compiler is not None
    probe = tmp_path / "framework_free_probe.cpp"
    probe.write_text(
        "#include \"cpu_compact_summary_reporting_ledger_v1.hpp\"\n"
        "#include <memory>\n"
        "int main() {\n"
        "  namespace ledger = "
        "eiel::native::cpu_compact_summary_reporting_ledger_v1;\n"
        "  namespace authority = "
        "eiel::cuda_architecture_b::compact_summary_reporting_authority_v1;\n"
        "  auto owner = std::make_unique<\n"
        "      ledger::CpuCompactSummaryReportingLedgerV1>(224, 512);\n"
        "  std::array<std::int64_t, 64> active_ids{};\n"
        "  std::array<std::int32_t, 225> agent_to_slot{};\n"
        "  std::array<float, 512> injury_by_slot{};\n"
        "  agent_to_slot.fill(-1);\n"
        "  for (std::int32_t id = 0; id < 64; ++id) {\n"
        "    active_ids[id] = id;\n"
        "    agent_to_slot[id] = id;\n"
        "  }\n"
        "  if (!owner->capacity_valid() || "
        "!owner->initialize_generation_from_slot_state_v1(\n"
        "          64, active_ids.data(), agent_to_slot.data(),\n"
        "          injury_by_slot.data(), 64, agent_to_slot.size())) {\n"
        "    return 1;\n"
        "  }\n"
        "  for (std::int64_t id = 64; id < 224; ++id) {\n"
        "    if (!owner->append_ordinary_birth_v1(id, id, 0.0F)) {\n"
        "      return 2;\n"
        "    }\n"
        "  }\n"
        "  if (owner->header().behavior_row_count != 224) {\n"
        "    return 3;\n"
        "  }\n"
        "  if (owner->append_ordinary_birth_v1(224, 224, 0.0F)) {\n"
        "    return 4;\n"
        "  }\n"
        "  if (owner->header().status != authority::kLedgerOverflowV1) {\n"
        "    return 5;\n"
        "  }\n"
        "  auto dense_owner = std::make_unique<\n"
        "      ledger::CpuCompactSummaryReportingLedgerV1>(224, 1024);\n"
        "  if (!dense_owner->capacity_valid()) {\n"
        "    return 6;\n"
        "  }\n"
        "  auto high_id_owner = std::make_unique<\n"
        "      ledger::CpuCompactSummaryReportingLedgerV1>(2, 4);\n"
        "  constexpr std::int64_t high_id = 1000000007LL;\n"
        "  const std::array<std::int64_t, 2> high_ids{7, high_id};\n"
        "  const std::array<std::int32_t, 2> high_slots{3, 1};\n"
        "  std::array<float, 4> high_injury{};\n"
        "  if (!high_id_owner->initialize_generation_from_active_slot_state_v1(\n"
        "          high_id + 1, high_ids.data(), high_slots.data(),\n"
        "          high_injury.data(), high_ids.size())) {\n"
        "    return 7;\n"
        "  }\n"
        "  if (high_id_owner->header().behavior_row_count != 2\n"
        "      || high_id_owner->behavior_rows()[1].agent_id != high_id) {\n"
        "    return 8;\n"
        "  }\n"
        "  return 0;\n"
        "}\n",
        encoding="ascii",
    )
    executable = tmp_path / "framework_free_probe"
    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fno-exceptions",
            "-fno-rtti",
            f"-I{ROOT / 'engine' / 'perf' / 'native'}",
            f"-I{ROOT / 'engine' / 'perf'}",
            str(probe),
            str(SOURCE),
            str(HOST_CANONICAL_SOURCE),
            "-o",
            str(executable),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    executed = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True
    )
    assert executed.returncode == 0, executed.stderr


def test_g2_behavior_and_lineage_digests_match_python_authority(
    native_ledger: ModuleType,
) -> None:
    receipt = native_ledger.run_g2_fixture()
    assert receipt["success"] is True
    for generation in receipt["generations"]:
        canonical = generation["canonical"]
        assert canonical["status"] == 0
        assert canonical["behavior_digest"].hex() == behavior_rows_digest_sha256(
            generation["behavior_rows"]
        )
        assert canonical["lineage_digest"].hex() == lineage_records_digest_sha256(
            generation["generation"], generation["lineage_records"]
        )


def test_action_birth_death_rows_and_b1_repeat_are_exact(
    native_ledger: ModuleType,
) -> None:
    receipt = native_ledger.run_g2_fixture()
    assert receipt["b1_repeat_exact"] is True
    generation_zero, generation_one = receipt["generations"]

    rows_zero = generation_zero["behavior_rows"]
    assert [row["agent_id"] for row in rows_zero] == [10, 11, 12]
    assert [row["cohort"] for row in rows_zero] == ["start", "start", "born"]
    assert rows_zero[0]["move_n"] == 2
    assert rows_zero[0]["alive_end"] == 0
    assert rows_zero[1]["attack"] == 1
    assert rows_zero[1]["alive_end"] == 1
    assert rows_zero[2]["gather"] == 1
    assert rows_zero[2]["eligible_for_selection"] == 0
    assert [row["agent_id"] for row in generation_zero["lineage_records"]] == [
        11,
        12,
    ]

    rows_one = generation_one["behavior_rows"]
    assert [row["agent_id"] for row in rows_one] == [11, 12]
    assert rows_one[0]["rest"] == 1
    assert rows_one[1]["gather"] == 2
    assert all(row["kept_next"] == 1 for row in rows_one)

    assert generation_zero["canonical"]["behavior"] == {
        "row_count": 3,
        "cohort_size": 2,
        "cohort_survivors": 1,
        "action_total": 4,
        "move_total": 2,
        "move_n": 2,
        "move_s": 0,
        "move_e": 0,
        "move_w": 0,
        "gather": 1,
        "rest": 0,
        "attack": 1,
    }


def test_retained_row_copy_uses_the_exact_cuda_reporting_layout(
    native_ledger: ModuleType,
) -> None:
    receipt = native_ledger.run_g2_fixture()
    for generation in receipt["generations"]:
        assert generation["retained_row_copied"] is True
        retained = np.frombuffer(
            generation["retained_row"], dtype=np.uint8
        ).reshape(1, 1, -1)
        projected = project_retained_reporting_authority(
            retained,
            first_generation=generation["generation"],
            row_capacity=4,
        )[0][0]
        assert projected["behavior_rows"] == generation["behavior_rows"]
        assert projected["lineage_records"] == generation["lineage_records"]
        assert projected["behavior_rows_digest_sha256"] == generation[
            "canonical"
        ]["behavior_digest"].hex()
        assert projected["lineage_records_digest_sha256"] == generation[
            "canonical"
        ]["lineage_digest"].hex()


def test_g2_lineage_metrics_continue_against_python_authority(
    native_ledger: ModuleType,
) -> None:
    receipt = native_ledger.run_g2_fixture()
    state = LineageMetricsState()
    for generation in receipt["generations"]:
        expected, _, _, state = compute_lineage_snapshot(
            generation=generation["generation"],
            agents=[
                _native_agent(record)
                for record in generation["lineage_records"]
            ],
            sim_cfg=SimpleNamespace(**LINEAGE_CONFIG),
            state=state,
        )
        actual = generation["canonical"]
        actual_lineage = actual["lineage"]
        assert actual_lineage["active_lineages"] == expected["active_lineages"]
        assert actual_lineage["dominant_lineage_id"] == expected[
            "dominant_lineage_id"
        ]
        assert actual_lineage["rare_persistent_lineages"] == expected[
            "rare_persistent_lineages"
        ]
        assert actual_lineage["collapse_streak"] == expected[
            "lineage_collapse_streak"
        ]
        assert actual_lineage["collapse_warning_code"] == (
            2 if expected["lineage_collapse_warning"] else 1
        )
        for native_name, python_name in (
            ("dominant_population_share", "dominant_lineage_population_share"),
            ("top3_population_share", "top3_lineage_population_share"),
            ("population_entropy", "lineage_population_entropy"),
            ("reproduction_entropy", "lineage_reproduction_entropy"),
            (
                "dominant_reproduction_share",
                "dominant_lineage_reproduction_share",
            ),
        ):
            _assert_float_bits_equal(
                actual_lineage[native_name], expected[python_name]
            )
        assert actual["lineage_state_after"] == {
            "collapse_streak": state.collapse_streak,
            "population_dominance_streak": state.population_dominance_streak,
            "reproduction_dominance_streak": (
                state.reproduction_dominance_streak
            ),
        }

    assert receipt["generations"][0]["canonical"]["lineage_state_after"] == {
        "collapse_streak": 1,
        "population_dominance_streak": 1,
        "reproduction_dominance_streak": 1,
    }
    assert receipt["generations"][1]["canonical"]["lineage_state_after"] == {
        "collapse_streak": 2,
        "population_dominance_streak": 2,
        "reproduction_dominance_streak": 2,
    }


def test_capacity_overflow_write_after_seal_and_failure_commit_contracts(
    native_ledger: ModuleType,
) -> None:
    receipt = native_ledger.run_failure_contracts()
    assert receipt["maximum_capacity_valid"] is True
    assert receipt["row_zero_rejected"] is True
    assert receipt["row_above_max_rejected"] is True
    assert receipt["slot_zero_rejected"] is True
    assert receipt["slot_above_max_rejected"] is True
    assert receipt["negative_lineage_state_rejected"] is True

    assert receipt["overflow_initialized"] is True
    assert receipt["overflow_first_birth"] is True
    assert receipt["overflow_rejected"] is True
    assert receipt["overflow_header_status"] == 102
    assert receipt["overflow_wrapper_status"] == "authority_failure"
    assert receipt["overflow_committed"] is False

    assert receipt["write_after_seal_prepared"] is True
    assert receipt["write_after_seal_committed_before"] is True
    assert receipt["write_after_seal_rejected"] is True
    assert receipt["write_after_seal_header_status"] == 109
    assert receipt["write_after_seal_committed_after"] is False
    assert receipt["write_after_seal_payload_clear"] is True
    assert receipt["write_after_seal_state_preserved"] is True

    assert receipt["canonical_failure_prepared"] is True
    assert receipt["canonical_failure_rejected"] is True
    assert receipt["canonical_failure_status"] == 18
    assert receipt["canonical_failure_payload_clear"] is True
    assert receipt["canonical_failure_not_committed"] is True
    assert receipt["canonical_retry_committed"] is True
    assert receipt["canonical_retry_state"] == {
        "collapse_streak": 1,
        "population_dominance_streak": 1,
        "reproduction_dominance_streak": 1,
    }
