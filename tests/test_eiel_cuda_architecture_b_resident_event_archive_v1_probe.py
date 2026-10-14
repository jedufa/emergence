from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from engine.perf.cuda_architecture_b_boundary import (
    EVENT_COLUMN_DTYPES,
    EVENT_COLUMN_NAMES,
)
from engine.perf.cuda_architecture_b_resident_window_runtime_v1 import (
    drain_event_window,
)
from tools.scripts import (
    eiel_cuda_architecture_b_resident_event_archive_v1_probe as probe,
)


def _python_oracle_entrypoint(
    columns: list[torch.Tensor],
    active_count: torch.Tensor,
    sequence_base: torch.Tensor,
    status: torch.Tensor,
) -> dict[str, object]:
    event_columns = dict(zip(EVENT_COLUMN_NAMES, columns, strict=True))
    bases = [int(value) for value in sequence_base.detach().to(device="cpu").tolist()]
    archives = probe._archive_prefixes(bases)
    prefix_lengths = [len(archive) for archive in archives]
    receipt = drain_event_window(
        event_columns,
        active_count=active_count,
        event_sequence_base=sequence_base,
        event_status=status,
        archives=archives,
    )
    chunks = []
    for cell, archive in enumerate(archives):
        appended = archive[prefix_lengths[cell] :]
        chunks.append(
            [[chunk[name] for name in EVENT_COLUMN_NAMES] for chunk in appended]
        )
    return {
        "schema_version": "test_python_oracle_proxy",
        "event_chunks": chunks,
        "active_count_before": receipt["active_count_before"],
        "event_sequence_base_before": receipt["event_sequence_base_before"],
        "event_sequence_base_after": receipt["event_sequence_base_after"],
        "host_archive_bytes_appended": receipt["host_archive_bytes_appended"],
        "source_columns_preserved": True,
    }


def _python_oracle_module() -> SimpleNamespace:
    module = SimpleNamespace()
    setattr(module, probe.ENTRYPOINT, _python_oracle_entrypoint)
    return module


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


def test_synthetic_bank_has_exact_22_column_contract() -> None:
    bank = probe._build_synthetic_bank("cpu")

    assert tuple(bank.columns) == EVENT_COLUMN_NAMES
    assert len(bank.columns) == 22
    assert bank.active_count.tolist() == list(probe.ACTIVE_COUNTS)
    assert bank.sequence_base.tolist() == list(probe.SEQUENCE_BASES)
    assert len(set(bank.active_count.tolist())) == 2
    assert bank.sequence_base.tolist() == [0, 0]
    for name in EVENT_COLUMN_NAMES:
        assert bank.columns[name].dtype == EVENT_COLUMN_DTYPES[name]
        assert tuple(bank.columns[name].shape) == (
            len(probe.ACTIVE_COUNTS),
            probe.PHYSICAL_CAPACITY,
        )
        assert bank.columns[name].is_contiguous()
    for cell, (count, base) in enumerate(
        zip(probe.ACTIVE_COUNTS, probe.SEQUENCE_BASES, strict=True)
    ):
        assert torch.equal(
            bank.columns["typed_lifecycle_event_sequence"][cell, :count],
            torch.arange(base, base + count, dtype=torch.int64),
        )


def test_tensor_pair_rejects_signed_zero_difference() -> None:
    expected = torch.tensor([0.0], dtype=torch.float32)
    actual = torch.tensor([-0.0], dtype=torch.float32)

    assert torch.equal(expected, actual)
    with pytest.raises(probe.ProbeMismatch, match="tensor bytes differ"):
        probe._record_tensor_pair(
            hashlib.sha256(),
            hashlib.sha256(),
            label="signed_zero",
            expected=expected,
            actual=actual,
        )


def test_device_comparison_accepts_exact_python_oracle_proxy() -> None:
    receipt = probe._run_device_comparison(_python_oracle_module(), "cpu")

    assert receipt["verdict"] == "pass"
    assert receipt["byte_exact"] is True
    assert receipt["active_counts"] == [2, 4]
    assert receipt["sequence_bases"] == [0, 0]
    assert receipt["actual_sha256"] == receipt["expected_sha256"]
    assert receipt["fail_closed_case_count"] == 2
    assert {case["case"] for case in receipt["fail_closed"]} == {
        "bad_sequence",
        "nonzero_status",
    }
    assert all(case["verdict"] == "pass" for case in receipt["fail_closed"])
    assert receipt["pointer_identity_checks"] == len(EVENT_COLUMN_NAMES) + 3
    assert receipt["tensor_comparisons"] >= len(EVENT_COLUMN_NAMES) * 4
    assert receipt["no_mutation_checks"] == 2 * 2 * (len(EVENT_COLUMN_NAMES) + 3)


def test_fail_closed_check_rejects_native_mutation() -> None:
    def mutating_entrypoint(
        columns: list[torch.Tensor],
        active_count: torch.Tensor,
        sequence_base: torch.Tensor,
        status: torch.Tensor,
    ) -> None:
        del columns, sequence_base, status
        active_count.zero_()
        raise RuntimeError("event status must be zero before drain")

    module = SimpleNamespace()
    setattr(module, probe.ENTRYPOINT, mutating_entrypoint)

    with pytest.raises(probe.ProbeMismatch, match="active_count bytes changed"):
        probe._run_fail_closed_case(
            module,
            "cpu",
            "nonzero_status",
            hashlib.sha256(),
            hashlib.sha256(),
        )


def test_run_probe_emits_devices_comparisons_hashes_and_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = tmp_path / "candidate.so"
    extension.write_bytes(b"immutable-extension")
    monkeypatch.setattr(probe, "_load_extension", lambda path: _python_oracle_module())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    receipt = probe.run_probe(extension)

    assert receipt["schema_version"] == probe.SCHEMA_VERSION
    assert (
        receipt["extension_sha256"]
        == hashlib.sha256(extension.read_bytes()).hexdigest()
    )
    assert receipt["devices"] == ["cpu"]
    assert receipt["comparisons"]["device_count"] == 1
    assert receipt["comparisons"]["fail_closed_case_count"] == 2
    assert (
        receipt["hashes"]["actual_combined_sha256"]
        == receipt["hashes"]["expected_combined_sha256"]
    )
    assert receipt["verdict"] == "pass"


def test_write_receipt_is_sorted_atomic_json(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "receipt.json"
    payload = {"verdict": "pass", "devices": ["cpu"]}

    probe._write_receipt(output, payload)

    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert not output.with_name(f".{output.name}.tmp").exists()
