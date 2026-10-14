from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tools.scripts.eiel_option_d_m17_runner_drive_noise_receipt_v1 import (
    M17ReceiptError,
    build_receipt,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_projection(path: Path, *, generation: int, metrics_np: int, state: int) -> None:
    payload = {
        "source_generation": generation,
        "source_tick": 0,
        "source_rng_sha256": f"{state:064x}",
        "source_projection_sha256": f"{generation:064x}",
        "projection": {
            "metadata": {
                "native_runtime_numpy_runner_state": {
                    "bit_generator": "PCG64",
                    "state": {"state": state, "inc": 11},
                    "has_uint32": 0,
                    "uinteger": 0,
                },
                "native_runtime_rng_manager_stream_indices": {
                    "metrics": {"np": metrics_np}
                },
            },
            "tensors": {
                "drive_noise_samples_by_tick_slot": {
                    "dtype": "float32",
                    "shape": [160, 384, 4],
                    "sha256": "a" * 64,
                },
                "drive_noise_samples_valid_by_tick_slot": {
                    "dtype": "bool",
                    "shape": [160, 384],
                    "sha256": "b" * 64,
                },
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)


def _write_sources(root: Path) -> tuple[Path, Path, Path]:
    kernels = root / "kernels.cu"
    direct = root / "direct.py"
    binding = root / "binding.cpp"
    kernels.write_text(
        "materialize_missing_drive_noise_v1(\n"
        "pcg64_numpy_standard_normal(rng);\n"
        "full_tick_outputs.drive_noise_draws[cell_tick];\n"
        "metrics_np_semantic_draws[cell] += value;\n",
        encoding="utf-8",
    )
    direct.write_text("direct resident allocator\n", encoding="utf-8")
    binding.write_text(
        "eiel_cuda_architecture_b_refill_drive_noise_window_v1\n",
        encoding="utf-8",
    )
    return kernels, direct, binding


def _gate(path: Path, *, host_sync: int = 0) -> Path:
    chunk = {
        "exact_cpu_policy_service_calls_inside_tick_loop": 0,
        "host_synchronizations_inside_loop": host_sync,
        "intermediate_host_synchronizations": 0,
        "intermediate_device_to_host_operations": 0,
        "intermediate_host_to_device_operations": 0,
        "aten_operations_inside_tick_loop": 0,
        "native_host_policy_service_used": False,
    }
    _write_json(
        path,
        {
            "ok": True,
            "status": "closed",
            "decision": "test-route",
            "dispatch": {
                "chunk_receipts": [chunk, dict(chunk)],
                "native_controller_receipt": {
                    "native_host_policy_service_used": False
                },
            },
        },
    )
    return path


def test_m17_receipt_binds_device_ownership_and_fails_closed_for_promotion(
    tmp_path: Path,
) -> None:
    gate = _gate(tmp_path / "gate.json")
    before = tmp_path / "before.json.gz"
    after = tmp_path / "after.json.gz"
    _write_projection(before, generation=2, metrics_np=100, state=123)
    _write_projection(after, generation=4, metrics_np=180, state=456)
    kernels, direct, binding = _write_sources(tmp_path)

    receipt = build_receipt(
        gate_path=gate,
        checkpoint_before_path=before,
        checkpoint_after_path=after,
        kernels_source=kernels,
        direct_source=direct,
        binding_source=binding,
    )

    assert receipt["passed"] is True
    assert receipt["operator_id"] == "runner.drive_noise.standard_normal"
    assert receipt["stream_id"] == "numpy.runner"
    assert receipt["ownership"]["zero_inside_loop_cpu_native_boundary"] is True
    assert receipt["ownership"]["direct_resident_host_refill_called"] is False
    assert receipt["checkpoint_boundary"]["runner_pcg64_state_changed"] is True
    assert receipt["checkpoint_boundary"]["metrics_np_semantic_draw_index_delta"] == 80
    assert receipt["draw_accounting"]["raw_pcg64_delta_available"] is False
    assert receipt["draw_accounting"]["promotion_qualified"] is False


def test_m17_receipt_rejects_nonzero_inside_loop_boundary(tmp_path: Path) -> None:
    gate = _gate(tmp_path / "gate.json", host_sync=1)
    before = tmp_path / "before.json.gz"
    after = tmp_path / "after.json.gz"
    _write_projection(before, generation=2, metrics_np=100, state=123)
    _write_projection(after, generation=4, metrics_np=180, state=456)
    kernels, direct, binding = _write_sources(tmp_path)

    receipt = build_receipt(
        gate_path=gate,
        checkpoint_before_path=before,
        checkpoint_after_path=after,
        kernels_source=kernels,
        direct_source=direct,
        binding_source=binding,
    )

    assert receipt["passed"] is False
    assert receipt["decision"] == "M17-REJECTED-RUNNER-DRIVE-NOISE-RECEIPT"
    assert receipt["ownership"]["zero_inside_loop_cpu_native_boundary"] is False


def test_m17_receipt_requires_runner_pcg64_projection_metadata(tmp_path: Path) -> None:
    gate = _gate(tmp_path / "gate.json")
    before = tmp_path / "before.json.gz"
    after = tmp_path / "after.json.gz"
    _write_projection(before, generation=2, metrics_np=100, state=123)
    _write_projection(after, generation=4, metrics_np=180, state=456)
    with gzip.open(before, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    del payload["projection"]["metadata"]["native_runtime_numpy_runner_state"]
    with gzip.open(before, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    kernels, direct, binding = _write_sources(tmp_path)

    with pytest.raises(M17ReceiptError, match="native_runtime_numpy_runner_state"):
        build_receipt(
            gate_path=gate,
            checkpoint_before_path=before,
            checkpoint_after_path=after,
            kernels_source=kernels,
            direct_source=direct,
            binding_source=binding,
        )
