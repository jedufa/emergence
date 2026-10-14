from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.perf import cuda_architecture_b_direct_resident_wdl_jungle_v1 as wdl_jungle


ROOT = Path(__file__).resolve().parents[1]
WDL_JUNGLE_SOURCE = (
    ROOT / "engine/perf/cuda_architecture_b_direct_resident_wdl_jungle_v1.py"
)
CPP_BINDING_SOURCE = (
    ROOT / "engine/perf/cuda_architecture_b_flattened_controller_v1_binding.cpp"
)
WDL_BINDING_SOURCE = (
    ROOT / "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_binding.cpp"
)


def _prepared_with_host_service(
    field_name: str,
    field_value: object,
) -> wdl_jungle.PreparedDirectResidentWdlJungleV1:
    """Build only the prefix needed to exercise the production admission gate."""

    configuration = SimpleNamespace(
        cuda_architecture_b_cpu_request_staging_enable=False,
        cuda_architecture_b_policy_birth_segment_refill_enable=False,
        cuda_architecture_b_cpython_gauss_segment_capacity=0,
    )
    setattr(configuration, field_name, field_value)
    source = SimpleNamespace(
        time_authority="absolute_tick_v1",
        patch_p2_config={"tw_rngv2_semantics_version": 2},
        configuration=configuration,
        generation_base=0,
        physical_window_generations=1,
    )
    patch = SimpleNamespace(source=source, arguments=())
    wdl = SimpleNamespace(
        patch=patch,
        initial_projection={"metadata": {"tick": 0}},
    )
    prepared = object.__new__(wdl_jungle.PreparedDirectResidentWdlJungleV1)
    object.__setattr__(prepared, "wdl", wdl)
    object.__setattr__(prepared, "jungle_state_tensors", ())
    object.__setattr__(prepared, "jungle_reporting_tensors", ())
    object.__setattr__(prepared, "jungle_config", {})
    object.__setattr__(prepared, "eed_state_tensors", ())
    object.__setattr__(prepared, "eed_config", {})
    object.__setattr__(prepared, "policy_window_identity_owner", None)
    return prepared


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        (
            "cuda_architecture_b_cpu_request_staging_enable",
            True,
        ),
        (
            "cuda_architecture_b_policy_birth_segment_refill_enable",
            True,
        ),
        (
            "cuda_architecture_b_cpython_gauss_segment_capacity",
            1,
        ),
    ),
    ids=("cpu-request-staging", "policy-birth-refill", "cpython-gaussian"),
)
def test_tick_native_wdl_jungle_eed_rejects_host_refill_services(
    field_name: str,
    field_value: object,
) -> None:
    prepared = _prepared_with_host_service(field_name, field_value)

    with pytest.raises(
        wdl_jungle.CudaArchitectureBDirectResidentWdlJungleV1Error,
        match="forbids CPU staging and host refill services",
    ):
        wdl_jungle._execute_direct_resident_wdl_jungle_eed_absolute_dispatch_v2(
            object(),
            prepared,
            span_ticks=1,
            start_tick=0,
            generation_base=0,
            absolute_start_next_tick=0,
            entrypoint_name="must_not_be_reached",
        )


def test_tick_native_host_refill_gate_precedes_cuda_entrypoint_lookup() -> None:
    source = WDL_JUNGLE_SOURCE.read_text(encoding="utf-8")
    gate_start = source.index("forbidden_host_services = {")
    gate_end = source.index("metadata = prepared.wdl.initial_projection", gate_start)
    gate = source[gate_start:gate_end]

    assert "cuda_architecture_b_cpu_request_staging_enable" in gate
    assert "cuda_architecture_b_policy_birth_segment_refill_enable" in gate
    assert "cuda_architecture_b_cpython_gauss_segment_capacity" in gate
    assert "raise _fail(" in gate
    assert "getattr(module, entrypoint_name, None)" not in gate


def test_cpp_rngv2_source_contract_requires_zero_tape_capacity() -> None:
    source = CPP_BINDING_SOURCE.read_text(encoding="utf-8")
    match = re.search(
        r"if\s*\(\s*ordinary_birth_config\.rng_semantics_version\s*==\s*2\s*\)\s*\{"
        r"(?P<body>.*?)\n\s*\}",
        source,
        flags=re.DOTALL,
    )
    assert match is not None, "missing C++ RNG V2 admission guard"
    body = match.group("body")
    assert "ordinary_birth_config.gauss_tape_capacity == 0" in body
    assert "ordinary_birth_config.policy_tape_capacity == 0" in body
    assert "TW RNG V2 forbids host-refilled Gaussian and policy tapes" in body


def test_cpp_rngv2_source_contract_rejects_exact_policy_callbacks() -> None:
    source = CPP_BINDING_SOURCE.read_text(encoding="utf-8")
    message = "TW RNG V2 tick-native production forbids every CPU refill or host policy callback"
    message_offset = source.index(message)
    gate_start = source.rfind("TORCH_CHECK(", 0, message_offset)
    gate_end = source.index(");", message_offset) + 2
    gate = source[gate_start:gate_end]

    assert "ordinary_birth_config.rng_semantics_version != 2" in gate
    for callback in (
        "!exact_cpu_policy_refill",
        "!hapb_ordinary_birth_commit",
        "!hapb_turnover_commit",
        "!policy_birth_segment_refill",
        "!gaussian_segment_refill",
    ):
        assert callback in gate
    assert message in gate

    wdl_binding = WDL_BINDING_SOURCE.read_text(encoding="utf-8")
    macro_start = wdl_binding.index(
        "#define EIEL_CUDA_ARCHITECTURE_B_EXACT_CPU_POLICY_CONTEXT_ALLOWED"
    )
    macro_end = wdl_binding.index(
        '#include "cuda_architecture_b_flattened_controller_v1_binding.cpp"',
        macro_start,
    )
    macro = wdl_binding[macro_start:macro_end]
    assert "EIEL_WDL_COMPILED_EXACT_CPU_POLICY_CONTEXT_ALLOWED_V1" in macro
    assert "current_flat_jungle_context_v4()" in macro
    assert "!eiel::cuda_architecture_b::flattened_jungle_controller_v4::" in macro
    assert ".enabled" in macro
