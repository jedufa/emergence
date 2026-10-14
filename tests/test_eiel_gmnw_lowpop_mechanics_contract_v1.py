from pathlib import Path

import pytest

from engine.perf import cuda_architecture_b_direct_resident_gmnw_checkpoint_v1 as codec
from engine.perf.cuda_architecture_b_direct_resident_gmnw_gru_v1 import (
    CudaArchitectureBDirectResidentGmnwGruV1Error,
    GmnwConfigurationV1,
    _validated_configuration,
)


ROOT = Path(__file__).resolve().parents[1]
PERF = ROOT / "engine" / "perf"


def test_low_population_mechanics_is_not_driven_by_reporting_flag() -> None:
    context = (
        PERF / "cuda_architecture_b_flattened_gmnw_controller_v1.hpp"
    ).read_text(encoding="utf-8")
    binding = (
        PERF / "cuda_architecture_b_flattened_wdl_controller_v2_binding.cpp"
    ).read_text(encoding="utf-8")
    kernels = (
        PERF / "cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"
    ).read_text(encoding="utf-8")

    assert "bool terminal_low_population_no_reinit;" in context
    assert 'exact_bool(config, "terminal_low_population_no_reinit")' in binding
    assert "gmnw_context.terminal_low_population_no_reinit," in kernels
    assert "gmnw_context.selection_metrics.enabled," not in kernels


def test_low_population_mechanics_configuration_is_exact_boolean() -> None:
    _validated_configuration(
        GmnwConfigurationV1(terminal_low_population_no_reinit=True)
    )
    with pytest.raises(
        CudaArchitectureBDirectResidentGmnwGruV1Error,
        match="exact boolean",
    ):
        _validated_configuration(
            GmnwConfigurationV1(
                terminal_low_population_no_reinit=1,  # type: ignore[arg-type]
            )
        )


def test_checkpoint_config_accepts_explicit_low_population_mechanics() -> None:
    config = {
        "enabled": True,
        "generation_base": 0,
        "generation_rows_per_cell": 4,
        "cue_ticks": 1,
        "consequence_ticks": 1,
        "grace_generation_count": 4,
        "severe_generation_count": 4,
        "grace_damage_fraction": 0.15,
        "severe_damage_fraction": 0.45,
        "lethal_damage_fraction": 1.0,
        "grace_reproduction_scale": 0.5,
        "control_matrix_enabled": True,
        "policy_arm_codes": (3, 6),
        "cue_visible_at_decision": (False, True),
        "gru_founder_count": (8, 8),
        "qualification_trace_enabled": False,
        "qualification_trace_output_ticks": 0,
        "qualification_decision_tape_enabled": False,
        "selection_metrics_enabled": True,
        "terminal_low_population_no_reinit": True,
        "venv_consequence_profile_codes": (0, 0),
    }

    observed, sel1 = codec._validated_prepared_config(config, path="config")

    assert sel1 is True
    assert observed["terminal_low_population_no_reinit"] is True
