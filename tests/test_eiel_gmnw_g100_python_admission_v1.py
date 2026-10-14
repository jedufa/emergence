from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.perf import cuda_architecture_b_direct_resident_gmnw_gru_v1 as gmnw
from engine.perf import cuda_architecture_b_direct_resident_gru_v3 as gru_v3


class _PreparationReached(RuntimeError):
    pass


@pytest.mark.parametrize("generation_count", (4, 6, 37, 99, 100))
def test_gmnw_preparation_admits_bounded_generation_range(
    monkeypatch: pytest.MonkeyPatch,
    generation_count: int,
) -> None:
    monkeypatch.setattr(gmnw, "_validated_capability", lambda module: {})
    monkeypatch.setattr(gmnw, "_load_witness", lambda path: object())
    monkeypatch.setattr(
        gmnw,
        "_shared_witness_pack",
        lambda witness, *, base_observation_size, device: object(),
    )
    monkeypatch.setattr(gru_v3, "_validated_capability", lambda module: {})

    def pass_base_admission(runner: object) -> None:
        raise _PreparationReached

    monkeypatch.setattr(gru_v3, "_gru_patch_envelope", pass_base_admission)

    with pytest.raises(_PreparationReached):
        gmnw.build_prepared_direct_resident_gmnw_gru_v1(
            object(),
            qualified_cuda_module=object(),
            logical_generation_count=generation_count,
        )


@pytest.mark.parametrize("generation_count", (3, 101))
def test_gmnw_preparation_rejects_unsupported_generation_counts(
    monkeypatch: pytest.MonkeyPatch,
    generation_count: int,
) -> None:
    monkeypatch.setattr(gmnw, "_validated_capability", lambda module: {})

    with pytest.raises(
        gmnw.CudaArchitectureBDirectResidentGmnwGruV1Error,
        match=r"requires generations in \[4,100\]",
    ):
        gmnw.build_prepared_direct_resident_gmnw_gru_v1(
            object(),
            qualified_cuda_module=object(),
            logical_generation_count=generation_count,
            cuda_device="cpu",
        )


def test_gmnw_execution_admits_exact_g100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"started": False}

    class RunControl:
        def manifest(self) -> dict[str, object]:
            if not state["started"]:
                return {
                    "run_id": "g100-admission",
                    "status": "initialized",
                    "run_started": False,
                    "sealed": False,
                }
            return {
                "run_id": "g100-admission",
                "status": "generation_complete",
                "requested_generations": 100,
                "active_generation": 99,
                "sealed": False,
                "awaiting_final_output_ack": True,
                "awaiting_cancellation_checkpoint": False,
                "cancel_requested": False,
            }

    def execute(control: RunControl, *arguments: object) -> dict[str, int]:
        assert arguments[4] == 100
        state["started"] = True
        return {"completed_generations": 100}

    module = SimpleNamespace(**{gmnw.GMNW_DIRECT_ENTRYPOINT_V1: execute})
    prepared = SimpleNamespace(
        gru_tensors=(),
        gru_config={},
        gmnw_tensors=(),
        gmnw_config={},
        patch=SimpleNamespace(arguments=()),
        initial_projection=SimpleNamespace(metadata={}),
    )
    monkeypatch.setattr(gmnw, "_validate_result", lambda raw: dict(raw))
    monkeypatch.setattr(gmnw, "_require_internal_status_clean", lambda prepared: None)
    monkeypatch.setattr(
        gmnw,
        "_attach_gmnw_selection_metrics_v1",
        lambda result, prepared: None,
    )

    result = gmnw.execute_direct_resident_gmnw_gru_v1(
        module,
        prepared,
        logical_generation_count=100,
        run_id="g100-admission",
        run_control=RunControl(),
    )

    assert result["completed_generations"] == 100
    assert result["cancelled_prefix"] is False


@pytest.mark.parametrize("generation_count", (3, 101))
def test_gmnw_execution_rejects_unsupported_generation_counts(
    generation_count: int,
) -> None:
    with pytest.raises(
        gmnw.CudaArchitectureBDirectResidentGmnwGruV1Error,
        match=r"requires generations in \[4,100\]",
    ):
        gmnw.execute_direct_resident_gmnw_gru_v1(
            object(),
            SimpleNamespace(),
            logical_generation_count=generation_count,
            run_id="unsupported",
        )
