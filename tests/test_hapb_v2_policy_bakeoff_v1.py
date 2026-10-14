from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

from engine.perf.hapb_v2.experimental import HapbV2ExperimentalDisabled
from engine.perf.hapb_v2.fixtures_v1 import tiny_single_row_fixture_v1
from engine.perf.hapb_v2.policy_bakeoff_v1 import (
    compare_logits_to_reference_v1,
    float32_bits_v1,
    run_gpu_bakeoff_v1,
)


ROOT = Path(__file__).resolve().parents[1]
BAKEOFF = ROOT / "tools/scripts/eiel_hapb_v2_policy_bakeoff_v1.py"


def test_compare_logits_to_reference_uses_frozen_fixture() -> None:
    fixture = tiny_single_row_fixture_v1()
    match = compare_logits_to_reference_v1(
        fixture.expected_logits, fixture.expected_logits
    )
    assert match["exact"] is True
    assert match["mismatch_count"] == 0
    assert match["first_mismatch"] is None
    mutated = fixture.expected_logits.copy()
    mutated[0, 0] = np.float32(mutated[0, 0] + 1.0)
    miss = compare_logits_to_reference_v1(mutated, fixture.expected_logits)
    assert miss["exact"] is False
    assert miss["mismatch_count"] >= 1
    assert miss["first_mismatch"]["row"] == 0
    assert miss["first_mismatch"]["action"] == 0
    assert miss["first_mismatch"]["expected_bits"] == float32_bits_v1(
        fixture.expected_logits[0, 0]
    )
    assert miss["first_mismatch"]["got_bits"] == float32_bits_v1(mutated[0, 0])
    assert miss["first_mismatch"]["got_bits"] != miss["first_mismatch"]["expected_bits"]


def test_gpu_bakeoff_driver_requires_experimental_flag() -> None:
    previous = os.environ.pop("EIEL_HAPB_V2_EXPERIMENTAL", None)
    try:
        with pytest.raises(HapbV2ExperimentalDisabled):
            run_gpu_bakeoff_v1(output=Path("unused.json"))
    finally:
        if previous is not None:
            os.environ["EIEL_HAPB_V2_EXPERIMENTAL"] = previous


def test_execute_gpu_cli_without_flag_exits_2(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("EIEL_HAPB_V2_EXPERIMENTAL", None)
    result = subprocess.run(
        [
            sys.executable,
            str(BAKEOFF),
            "--execute-gpu",
            "--output",
            str(tmp_path / "receipt.json"),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "EIEL_HAPB_V2_EXPERIMENTAL=1" in result.stderr
    assert not (tmp_path / "receipt.json").exists()


def test_execute_gpu_cli_is_not_a_stub() -> None:
    source = BAKEOFF.read_text()
    assert "run_gpu_bakeoff_v1(" in source
    assert "return 3" not in source
    assert "this CPU prep invocation will not launch kernels" not in source
    driver = (ROOT / "engine/perf/hapb_v2/policy_bakeoff_v1.py").read_text()
    assert "compile_experimental_extension_v1" in driver
    assert "compare_logits_to_reference_v1" in driver
    assert "eiel_hapb_v2_nx64_policy_logits_experimental_v1" in driver
    assert "isolated_v1_nx1_device_helper" in driver
    assert "compare_logits_to_reference_v1(nx64_cpu, nx1_cpu)" in driver
