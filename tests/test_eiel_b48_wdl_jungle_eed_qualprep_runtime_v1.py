from __future__ import annotations

from tools.scripts import (
    eiel_b48_wdl_jungle_eed_qualprep_runtime_v1 as subject,
)
from tools.scripts import eiel_b48_non_eed_wdl_jungle_qualprep_runtime_v1 as base


def test_eed_runtime_wrapper_requires_the_combined_owner(monkeypatch) -> None:
    observed: list[str] = []

    def fake_main(argv: list[str]) -> int:
        observed[:] = argv
        return 17

    monkeypatch.setattr(subject._base, "main", fake_main)

    assert subject.main(["--cells", "2"]) == 17
    assert observed == ["--cells", "2", "--eed-overlay"]


def test_execution_receipt_retains_eed_native_ownership() -> None:
    summary = base._execution_summary(
        {
            "wdl_jungle_native_owner": True,
            "wdl_native_controller_abi_version": "wdl_v3",
            "jungle_native_controller_abi_version": "jungle_v4",
            "eed_native_owner": True,
            "eed_native_controller_abi_version": "eed_v1",
            "native_run_control_v2": {"status": "generation_complete"},
        }
    )

    assert summary["eed_native_owner"] is True
    assert summary["eed_native_controller_abi_version"] == "eed_v1"
