from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.scripts import (
    eiel_cuda_architecture_b_resident_window_refill_v1_probe as probe,
)


def test_options_require_existing_extension_and_json_output(tmp_path: Path) -> None:
    extension = tmp_path / "candidate.so"
    extension.write_bytes(b"extension")
    output = tmp_path / "receipts" / "probe.json"
    namespace = probe.build_parser().parse_args(
        ["--extension", str(extension), "--output", str(output)]
    )

    options = probe._options_from_namespace(namespace)

    assert options.extension == extension.resolve()
    assert options.output == output.resolve()

    namespace.output = tmp_path / "probe.txt"
    with pytest.raises(ValueError, match="must end in .json"):
        probe._options_from_namespace(namespace)

    namespace.extension = tmp_path / "missing.so"
    with pytest.raises(ValueError, match="existing .so"):
        probe._options_from_namespace(namespace)


def test_write_receipt_is_sorted_atomic_json(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "receipt.json"
    payload = {"verdict": "pass", "matrix_counts": {"total_cases": 48}}

    probe._write_receipt(output, payload)

    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert not output.with_name(f".{output.name}.tmp").exists()


def test_matrix_counts_cover_all_required_modes() -> None:
    counts = probe._matrix_counts()

    assert counts["gaussian_cases"] == 28
    assert counts["policy_cases"] == 20
    assert counts["policy_zero_std_cases"] == 8
    assert counts["total_cases"] == 48
    assert {case.name for case in probe.POLICY_CASES} == {
        "disabled",
        "dense",
        "sparse",
        "dense_zero_std",
        "sparse_zero_std",
    }


def test_byte_exact_comparison_rejects_signed_zero_difference() -> None:
    expected = torch.tensor([0.0], dtype=torch.float32)
    actual = torch.tensor([-0.0], dtype=torch.float32)

    assert torch.equal(expected, actual)
    with pytest.raises(probe.ProbeMismatch, match="tensor bytes differ"):
        probe._require_byte_exact(expected, actual, label="signed_zero")


def test_full_matrix_accepts_exact_python_oracle_proxy() -> None:
    from engine.perf.cuda_architecture_b_child_policy import (
        prepare_fixed_mlp_turnover_child_policy_tape,
    )
    from engine.perf.cuda_architecture_b_lifecycle import (
        pack_cpython_random_state_cuda,
    )

    def gaussian(words: torch.Tensor, index: int, capacity: int):
        state = (
            3,
            tuple(int(value) for value in words.tolist()) + (index,),
            None,
        )
        prepared = pack_cpython_random_state_cuda(
            state,
            device="cpu",
            gauss_tape_raw_capacity=capacity,
        )
        return {
            **{field: prepared[field] for field in probe.GAUSSIAN_TENSOR_FIELDS},
            "source_state_preserved": True,
            "staging_post_index": probe._expected_post_index(state, capacity),
        }

    def policy(
        state: torch.Tensor,
        episodes: int,
        width: int,
        enabled: bool,
        mode: str,
        rate: float,
        std: float,
    ):
        prepared = prepare_fixed_mlp_turnover_child_policy_tape(
            state,
            episode_capacity=episodes,
            row_width=width,
            device="cpu",
            policy_weight_mutation_enable=enabled,
            policy_weight_mutation_mode=mode,
            policy_weight_mutation_rate=rate,
            policy_weight_mutation_std=std,
        )
        return {**prepared, "source_state_preserved": True}

    module = SimpleNamespace()
    setattr(module, probe.GAUSSIAN_ENTRYPOINT, gaussian)
    setattr(module, probe.POLICY_ENTRYPOINT, policy)

    gaussian_receipt = probe._run_gaussian_matrix(module)
    policy_receipt = probe._run_policy_matrix(module)

    assert gaussian_receipt["case_count"] == 28
    assert gaussian_receipt["expected_sha256"] == gaussian_receipt["actual_sha256"]
    assert policy_receipt["case_count"] == 20
    assert policy_receipt["expected_sha256"] == policy_receipt["actual_sha256"]
