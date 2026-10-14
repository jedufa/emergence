from __future__ import annotations

from tools.scripts import eiel_wrc_resident_gpu_gate as gate


def test_gate_blocks_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(gate, "CUDA_AVAILABLE", lambda: False)

    receipt = gate.gate_receipt(mode="representative")

    assert receipt["decision"] == "BLOCKED"
    assert receipt["reason"] == "cuda_unavailable"
    assert receipt["cuda_available"] is False
    assert "physical B48" in receipt["no_claims"]


def test_gate_is_ready_only_when_cuda_is_present(monkeypatch) -> None:
    monkeypatch.setattr(gate, "CUDA_AVAILABLE", lambda: True)

    receipt = gate.gate_receipt(mode="representative")

    assert receipt["decision"] == "READY_FOR_EXACT_ADMISSION"
    assert receipt["reason"] is None
