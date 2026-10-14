from pathlib import Path

import pytest

from engine.authoritative_state.errors import AuthoritativeStateContractError
from engine.authoritative_state.runtime_abi import (
    resolve_prepared_direct_resident_trajectory_v1,
)
from engine.perf.cuda_architecture_b_cross_path_resident_overlay_v1 import (
    PreparedCrossPathDirectResidentPatchMulticellV1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BINDING = (
    PROJECT_ROOT
    / "engine/perf/cuda_architecture_b_flattened_controller_v1_binding.cpp"
)
CROSS_PATH_ENTRYPOINT = (
    "eiel_cuda_architecture_b_execute_direct_resident_cross_path_windows_v1"
)
DIRECT_ENTRYPOINT = "eiel_cuda_architecture_b_execute_direct_resident_windows_v1"


def test_production_binding_does_not_export_standalone_cross_path_execution() -> None:
    source = BINDING.read_text(encoding="utf-8")

    assert CROSS_PATH_ENTRYPOINT not in source
    assert f'"{DIRECT_ENTRYPOINT}"' in source


def test_authoritative_runtime_abi_rejects_cross_path_wrapper() -> None:
    retired = object.__new__(PreparedCrossPathDirectResidentPatchMulticellV1)

    with pytest.raises(
        AuthoritativeStateContractError,
        match="sanctioned typed wrapper",
    ):
        resolve_prepared_direct_resident_trajectory_v1(retired)
