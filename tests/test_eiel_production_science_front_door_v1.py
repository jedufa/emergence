from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from engine.campaigns.artifact_lifecycle import lifecycle_profile_contract_v1
from engine.campaigns.baseline_v1 import BASELINE_PROFILE_ID
from engine.campaigns.canopy_supervision_v1 import (
    CANOPY_RELAY_PROCESS_BUDGET_V1,
    start_canopy_relays_v1,
)
from engine.campaigns.canonical import canonical_sha256
from engine.qrr.production_front_door_v1 import (
    PRODUCTION_SCIENCE_REQUEST_SCHEMA_VERSION,
    PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION,
    ProductionScienceFrontDoorError,
    _SealedQrrInvocationError,
    _start_canopy_relays_v1,
    _validate_acs_materialization_v1,
    activate_qrr_v1,
    execute_science_request_v1,
    execute_science_request_v2,
    materialize_qrr_experiment_v1,
    validate_production_science_request_v1,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / (
    "_spec/campaigns/v1/"
    "wdl_jungle_eed_gv2_birth_w32p256c512_b48g200t160cuda_allocation_diagnostic.yaml"
)
ACS0_REQUEST = ROOT / (
    "_spec/production_science/v1/"
    "acs0_b48_w32_tick480_instrumentation.json"
)
BASELINE_TEST_DEVIATIONS = (
    "evolution.full_evolution",
    "runtime.production_cuda",
    "science.wrc_variability",
    "time.absolute_native_tick",
    "checkpoint.native_streamed",
    "checkpoint.acs_two_slot",
    "artifacts.lifecycle_contract",
    "authority.generation_runtime",
    "authority.device_fallback",
)


def _request() -> dict:
    return {
        "schema_version": PRODUCTION_SCIENCE_REQUEST_SCHEMA_VERSION,
        "request_id": "matrix-six-by-eight",
        "world_width": 32,
        "batch_cells": 48,
        "total_ticks": 49,
        "checkpoint_ticks": [17, 33, 49],
        "seeds": list(range(1001, 1009)),
        "configurations": [
            {
                "id": f"config-{index}",
                "controls": {
                    "env_cfg.move_cost": 0.1 + index * 0.05,
                    "env_cfg.jungle_predator_base_risk": 0.1 + index * 0.02,
                    "env_cfg.reproduce_try_prob": 0.2 + index * 0.05,
                },
            }
            for index in range(6)
        ],
        "deliberate_restore_ticks": [17, 33],
        "observability": {"living_lineage_canopy": True},
    }


def _request_v2() -> dict:
    """Build a Baseline-declared CPU fixture for front-door behavior tests."""

    request = _request()
    request["schema_version"] = PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION
    request["acs_checkpoint_slots"] = 0
    request["artifact_lifecycle"] = lifecycle_profile_contract_v1(
        "SCIENCE_COMPACT",
        total_max_bytes=1_000_000,
        checkpoint_schedule=request["checkpoint_ticks"],
        checkpoint_authority="absolute_tick",
    )
    request["baseline"] = {
        "profile_id": BASELINE_PROFILE_ID,
        "intent": "development",
        "deviations": [
            {
                "requirement_id": requirement_id,
                "reason": "bounded CPU fixture intentionally differs from Baseline",
            }
            for requirement_id in BASELINE_TEST_DEVIATIONS
        ],
    }
    return request


def _resolution_v2(request: dict, release_root: Path) -> dict:
    core = {
        "schema_version": "eiel_active_qrr_resolution_receipt_v1",
        "status": "PASS",
        "release_id": release_root.name,
        "manifest_sha256": "a" * 64,
        "registry_sha256": "b" * 64,
        "science_request_sha256": request["request_sha256"],
        "qualification_evidence_binding_sha256": "c" * 64,
        "manual_internal_decisions_required": 0,
        "reasons": [],
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _resolution_v1(request: dict, release_root: Path) -> dict:
    normalized_request = validate_production_science_request_v1(request)
    core = {
        "schema_version": "eiel_active_qrr_resolution_receipt_v1",
        "status": "PASS",
        "release_id": release_root.name,
        "manifest_sha256": "a" * 64,
        "registry_sha256": "b" * 64,
        "science_request_sha256": canonical_sha256(normalized_request),
        "qualification_evidence_binding_sha256": "c" * 64,
        "manual_internal_decisions_required": 0,
        "reasons": [],
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _v2_preflight(plan_sha256: str) -> dict:
    return {
        "schema_version": "eiel_campaign_preflight_v1",
        "plan_sha256": plan_sha256,
        "ready_for_execution": True,
        "execution_route": {},
    }


def _release_root(tmp_path: Path, *, default_move_cost: float | None = None) -> Path:
    root = tmp_path / "qrr-test"
    destination = root / "payload/spec/base_spec.yaml"
    destination.parent.mkdir(parents=True)
    source = BASE.read_text(encoding="utf-8")
    if default_move_cost is not None:
        source = source.replace(
            "parameters:\n",
            f"parameters:\n  env_cfg.move_cost: {default_move_cost}\n",
            1,
        )
    destination.write_text(source, encoding="utf-8")
    return root


def test_science_front_door_accepts_exact_six_by_eight_without_internal_ids(
    tmp_path: Path,
) -> None:
    request = validate_production_science_request_v1(_request())
    experiment = materialize_qrr_experiment_v1(
        request,
        release_root=_release_root(tmp_path),
    )

    assert request["batch_cells"] == 48
    assert len(experiment["configurations"]) == 6
    assert experiment["seeds"] == {"task": list(range(1001, 1009))}
    assert experiment["total_ticks"] == 49
    assert experiment["checkpoint_ticks"] == [17, 33, 49]
    assert experiment["deliberate_restore_ticks"] == [17, 33]
    assert "release_id" not in experiment
    assert "route" not in experiment


def test_science_front_door_binds_default_off_acs_checkpoint_selection(
    tmp_path: Path,
) -> None:
    default_request = validate_production_science_request_v1(_request())
    assert "acs_capture_instrumentation" not in default_request
    assert "acs_checkpoint_slots" not in default_request

    instrumentation_request = _request()
    instrumentation_request["acs_capture_instrumentation"] = True
    instrumentation = validate_production_science_request_v1(
        instrumentation_request
    )
    instrumentation_experiment = materialize_qrr_experiment_v1(
        instrumentation,
        release_root=_release_root(tmp_path / "instrumentation"),
    )
    assert instrumentation_experiment["acs_capture_instrumentation"] is True
    assert "acs_checkpoint_slots" not in instrumentation_experiment

    staging_request = _request()
    staging_request["acs_checkpoint_slots"] = 2
    staging = validate_production_science_request_v1(staging_request)
    staging_experiment = materialize_qrr_experiment_v1(
        staging,
        release_root=_release_root(tmp_path / "staging"),
    )
    assert staging_experiment["acs_capture_instrumentation"] is True
    assert staging_experiment["acs_checkpoint_slots"] == 2


def test_checked_in_acs0_request_is_exact_production_shaped_b48() -> None:
    request = validate_production_science_request_v1(
        json.loads(ACS0_REQUEST.read_text(encoding="utf-8"))
    )

    assert request["batch_cells"] == 48
    assert request["world_width"] == 32
    assert request["total_ticks"] == 480
    assert request["checkpoint_ticks"] == [480]
    assert request["deliberate_restore_ticks"] == []
    assert request["acs_capture_instrumentation"] is True
    assert "acs_checkpoint_slots" not in request
    assert len(request["configurations"]) == 6
    assert len(request["seeds"]) == 8


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("acs_capture_instrumentation", 1, "exact boolean"),
        ("acs_capture_instrumentation", "true", "exact boolean"),
        ("acs_checkpoint_slots", True, "zero, one, two, or three"),
        ("acs_checkpoint_slots", 4, "zero, one, two, or three"),
    ),
)
def test_science_front_door_rejects_malformed_acs_checkpoint_selection(
    field: str, value: object, message: str
) -> None:
    request = _request()
    request[field] = value

    with pytest.raises(ProductionScienceFrontDoorError, match=message):
        validate_production_science_request_v1(request)


def test_science_front_door_rejects_silent_acs_route_downgrade() -> None:
    request = validate_production_science_request_v1(
        {**_request(), "acs_capture_instrumentation": True}
    )
    with pytest.raises(
        ProductionScienceFrontDoorError, match="ACS_ROUTE_NOT_MATERIALIZED"
    ):
        _validate_acs_materialization_v1(
            {"execution": {"total_ticks": 49}},
            request=request,
        )

    _validate_acs_materialization_v1(
        {
            "execution": {
                "total_ticks": 49,
                "acs_capture_instrumentation": True,
            }
        },
        request=request,
    )


def test_science_front_door_fails_closed_on_matrix_shape_or_schedule() -> None:
    request = _request()
    request["batch_cells"] = 47
    with pytest.raises(ProductionScienceFrontDoorError, match="crossed with seeds"):
        validate_production_science_request_v1(request)

    request = _request()
    request["checkpoint_every_ticks"] = 5
    with pytest.raises(ProductionScienceFrontDoorError, match="exactly one"):
        validate_production_science_request_v1(request)


def test_science_front_door_defaults_canopy_and_rejects_malformed_observability() -> (
    None
):
    request = _request()
    del request["observability"]
    assert validate_production_science_request_v1(request)["observability"] == {
        "living_lineage_canopy": True
    }

    request["observability"] = {"living_lineage_canopy": "yes"}
    with pytest.raises(ProductionScienceFrontDoorError, match="must be boolean"):
        validate_production_science_request_v1(request)

    request["observability"] = {"unknown_projection": True}
    with pytest.raises(ProductionScienceFrontDoorError, match="unknown_projection"):
        validate_production_science_request_v1(request)


def test_default_science_configuration_is_derived_from_the_sealed_base(
    tmp_path: Path,
) -> None:
    request = _request()
    request["configurations"] = []
    request["seeds"] = list(range(48))
    experiment = materialize_qrr_experiment_v1(
        request,
        release_root=_release_root(tmp_path, default_move_cost=0.05),
    )

    assert experiment["configurations"] == [
        {"id": "default", "controls": {"env_cfg.move_cost": 0.05}}
    ]


def test_default_science_configuration_requires_explicit_sealed_value(
    tmp_path: Path,
) -> None:
    request = _request()
    request["configurations"] = []
    request["seeds"] = list(range(48))

    with pytest.raises(
        ProductionScienceFrontDoorError, match="explicit sealed default"
    ):
        materialize_qrr_experiment_v1(
            request,
            release_root=_release_root(tmp_path),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("checkpoint_ticks", [33, 17, 49]),
        ("checkpoint_ticks", [17, 17, 49]),
        ("checkpoint_ticks", [0, 33, 49]),
        ("checkpoint_ticks", [17, 33, 50]),
        ("deliberate_restore_ticks", [33, 17]),
        ("deliberate_restore_ticks", [17, 17]),
        ("deliberate_restore_ticks", [18, 33]),
        ("deliberate_restore_ticks", [17, 49]),
    ),
)
def test_science_front_door_rejects_invalid_public_schedules(
    field: str, value: list[int]
) -> None:
    request = _request()
    request[field] = value

    with pytest.raises(ProductionScienceFrontDoorError):
        validate_production_science_request_v1(request)


def test_science_front_door_rejects_duplicate_configuration_ids() -> None:
    request = _request()
    request["configurations"][1]["id"] = request["configurations"][0]["id"]

    with pytest.raises(ProductionScienceFrontDoorError, match="ids must be unique"):
        validate_production_science_request_v1(request)


def test_active_qrr_registry_activation_supersedes_prior_release_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qrr_root = tmp_path / "qrr"
    first = qrr_root / "qrr1-first"
    second = qrr_root / "qrr1-second"
    first.mkdir(parents=True)
    second.mkdir()
    releases = {
        first: SimpleNamespace(
            release_id=first.name,
            payload=lambda: {"manifest_sha256": "a" * 64},
        ),
        second: SimpleNamespace(
            release_id=second.name,
            payload=lambda: {"manifest_sha256": "b" * 64},
        ),
    }
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1.load_qualified_runtime_release_v1",
        lambda path: releases[path],
    )
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1._require_science_release_authority_v1",
        lambda _release: None,
    )

    first_registry = activate_qrr_v1(first, qrr_root=qrr_root)
    second_registry = activate_qrr_v1(second, qrr_root=qrr_root)

    assert first_registry["active_release_id"] == first.name
    assert second_registry["active_release_id"] == second.name
    rows = {row["release_id"]: row for row in second_registry["releases"]}
    assert rows[first.name]["status"] == "superseded"
    assert rows[first.name]["superseded_by"] == second.name
    assert rows[second.name]["status"] == "active"
    assert not (qrr_root / "active_qrr_registry_v1.json.tmp").exists()


def test_active_qrr_registry_rejects_tampered_previous_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qrr_root = tmp_path / "qrr"
    release_root = qrr_root / "qrr1-current"
    release_root.mkdir(parents=True)
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1.load_qualified_runtime_release_v1",
        lambda _path: SimpleNamespace(
            release_id=release_root.name,
            payload=lambda: {"manifest_sha256": "a" * 64},
        ),
    )
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1._require_science_release_authority_v1",
        lambda _release: None,
    )
    registry = activate_qrr_v1(release_root, qrr_root=qrr_root)
    registry["active_release_id"] = "qrr1-tampered"
    (qrr_root / "active_qrr_registry_v1.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )

    with pytest.raises(ProductionScienceFrontDoorError, match="registry is malformed"):
        activate_qrr_v1(release_root, qrr_root=qrr_root)


def test_science_front_door_starts_one_detached_canopy_relay_per_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict]] = []

    class _Process:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def terminate(self) -> None:
            raise AssertionError("healthy relay was terminated")

    def _popen(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return _Process(4100 + len(calls))

    monkeypatch.setattr("engine.qrr.production_front_door_v1.subprocess.Popen", _popen)
    run_root = tmp_path / "run"
    receipt = _start_canopy_relays_v1(
        plan={
            "plan_sha256": "a" * 64,
            "observability": {"living_lineage_canopy": True},
            "cells": [{"cell_id": "cell-a"}, {"cell_id": "cell-b"}],
        },
        run_root=run_root,
        relay_root=tmp_path / "live/canopy-relays",
        repository_root=ROOT,
        python_executable=Path(sys.executable),
    )

    assert receipt["state"] == "watching"
    assert [row["cell_id"] for row in receipt["relays"]] == ["cell-a", "cell-b"]
    assert len(calls) == 2
    assert all(call[1]["start_new_session"] is True for call in calls)
    assert all("--poll-seconds" in call[0] for call in calls)
    assert (run_root / "canopy_relay_supervision.json").is_file()

    calls.clear()
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1._relay_process_matches_v1",
        lambda _pid, output_root: True,
    )
    reused = _start_canopy_relays_v1(
        plan={
            "plan_sha256": "a" * 64,
            "observability": {"living_lineage_canopy": True},
            "cells": [{"cell_id": "cell-a"}, {"cell_id": "cell-b"}],
        },
        run_root=run_root,
        relay_root=tmp_path / "live/canopy-relays",
        repository_root=ROOT,
        python_executable=Path(sys.executable),
    )
    assert calls == []
    assert all(row["reused"] is True for row in reused["relays"])


def test_canopy_relay_supervision_records_degradation_at_bounded_logical_cell_scale(
    tmp_path: Path,
) -> None:
    receipt = start_canopy_relays_v1(
        plan={
            "plan_sha256": "a" * 64,
            "cells": [
                {"cell_id": f"cell-{index:03d}"}
                for index in range(CANOPY_RELAY_PROCESS_BUDGET_V1 + 1)
            ],
        },
        run_root=tmp_path / "run",
        relay_root=tmp_path / "live/canopy-relays",
        repository_root=ROOT,
        python_executable=Path(sys.executable),
        enabled=True,
        popen_fn=lambda *_args, **_kwargs: pytest.fail(
            "oversized logical bundle must not spawn relay processes"
        ),
    )
    assert receipt["state"] == "observability_degraded"
    assert receipt["authoritative_runtime_blocked"] is False
    assert receipt["simulation_state_mutation"] is False
    assert receipt["relays"] == []
    assert receipt["degradation_details"] == {
        "requested_cell_count": CANOPY_RELAY_PROCESS_BUDGET_V1 + 1,
        "relay_process_budget": CANOPY_RELAY_PROCESS_BUDGET_V1,
        "relay_start_policy": "bounded_observability_degradation_v1",
    }


def test_science_front_door_preflights_then_starts_relays_before_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request_v2()), encoding="utf-8")
    release_root = tmp_path / "qrr/qrr1-active"
    release_root.mkdir(parents=True)
    run_root = tmp_path / "run"
    events: list[str] = []

    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1.resolve_active_qrr_v2",
        lambda request, qrr_root: (
            release_root,
            _resolution_v2(request, release_root),
        ),
    )
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1.materialize_qrr_experiment_v2",
        lambda _request, release_root: {"schema_version": "eiel_qrr_experiment_v1"},
    )

    def _invoke(argv):
        command = argv[1]
        events.append(command)
        if command == "preflight":
            (run_root / "execution_plan.json").write_text(
                json.dumps({"plan_sha256": "c" * 64}), encoding="utf-8"
            )
            return {
                "ok": True,
                "command": command,
                "preflight": _v2_preflight("c" * 64),
            }
        return {"ok": True, "command": command}

    def _start(**_kwargs):
        events.append("relays")
        return {"state": "watching", "relays": []}

    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1._invoke_sealed_qrr_v1", _invoke
    )
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1._start_canopy_relays_v1", _start
    )

    result = execute_science_request_v2(
        "run", request_path, qrr_root=tmp_path / "qrr", run_root=run_root
    )

    assert result["ok"] is True
    assert events == ["preflight", "relays", "run"]
    assert result["canopy_relay_supervision"]["state"] == "watching"


def test_science_front_door_records_canopy_degradation_without_gating_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request_v2()), encoding="utf-8")
    release_root = tmp_path / "qrr/qrr1-active"
    release_root.mkdir(parents=True)
    run_root = tmp_path / "run"
    events: list[str] = []
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1.resolve_active_qrr_v2",
        lambda request, qrr_root: (
            release_root,
            _resolution_v2(request, release_root),
        ),
    )
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1.materialize_qrr_experiment_v2",
        lambda _request, release_root: {"schema_version": "eiel_qrr_experiment_v1"},
    )

    def _invoke(argv):
        command = argv[1]
        events.append(command)
        if command == "preflight":
            (run_root / "execution_plan.json").write_text(
                json.dumps({"plan_sha256": "c" * 64}), encoding="utf-8"
            )
            return {
                "ok": True,
                "command": command,
                "preflight": _v2_preflight("c" * 64),
            }
        return {"ok": True, "command": command}

    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1._invoke_sealed_qrr_v1", _invoke
    )
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1._start_canopy_relays_v1",
        lambda **_kwargs: (_ for _ in ()).throw(
            ProductionScienceFrontDoorError("CANOPY_BLOCKED: relay unavailable")
        ),
    )

    result = execute_science_request_v2(
        "run", request_path, qrr_root=tmp_path / "qrr", run_root=run_root
    )

    assert result["ok"] is True
    assert events == ["preflight", "run"]
    supervision = result["canopy_relay_supervision"]
    assert supervision["state"] == "observability_degraded"
    assert supervision["authoritative_runtime_blocked"] is False
    assert supervision["simulation_state_mutation"] is False
    assert supervision["relays"] == []
    assert (run_root / "canopy_relay_supervision.json").is_file()
    assert not (run_root / "canopy_source_terminal.json").exists()


def test_science_front_door_marks_canopy_interrupted_when_launch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request_v2()), encoding="utf-8")
    release_root = tmp_path / "qrr/qrr1-active"
    release_root.mkdir(parents=True)
    run_root = tmp_path / "run"
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1.resolve_active_qrr_v2",
        lambda request, qrr_root: (
            release_root,
            _resolution_v2(request, release_root),
        ),
    )
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1.materialize_qrr_experiment_v2",
        lambda _request, release_root: {"schema_version": "eiel_qrr_experiment_v1"},
    )

    def _invoke(argv):
        if argv[1] == "preflight":
            (run_root / "execution_plan.json").write_text(
                json.dumps({"plan_sha256": "c" * 64}), encoding="utf-8"
            )
            return {"ok": True, "preflight": _v2_preflight("c" * 64)}
        raise RuntimeError("sealed launch failed")

    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1._invoke_sealed_qrr_v1", _invoke
    )
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1._start_canopy_relays_v1",
        lambda **_kwargs: {
            "state": "watching",
            "relays": [{"cell_id": "cell-a", "pid": 4100}],
        },
    )

    with pytest.raises(RuntimeError, match="sealed launch failed"):
        execute_science_request_v2(
            "run", request_path, qrr_root=tmp_path / "qrr", run_root=run_root
        )

    terminal = json.loads((run_root / "canopy_source_terminal.json").read_text())
    assert terminal["state"] == "source_interrupted"
    assert terminal["reason"] == "RuntimeError: sealed launch failed"
    assert len(terminal["record_sha256"]) == 64


def test_v1_run_is_allowed_only_as_exact_bound_nonterminal_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "request.json"
    request = _request()
    request_path.write_text(json.dumps(request), encoding="utf-8")
    release_root = tmp_path / "qrr/qrr1-active"
    release_root.mkdir(parents=True)
    run_root = tmp_path / "run"
    resolution = _resolution_v1(request, release_root)
    events: list[str] = []

    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1.resolve_active_qrr_v1",
        lambda _request, qrr_root: (release_root, resolution),
    )
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1._load_bound_science_resolution_v1",
        lambda **_kwargs: (release_root, resolution),
    )
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1.materialize_qrr_experiment_v1",
        lambda _request, release_root: {
            "schema_version": "eiel_qrr_experiment_v1"
        },
    )

    def _invoke(argv):
        command = argv[1]
        events.append(command)
        if command == "materialize":
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "execution_plan.json").write_text(
                json.dumps({"plan_sha256": "c" * 64}), encoding="utf-8"
            )
            (run_root / "qrr_run_binding.json").write_text(
                json.dumps(
                    {
                        "schema_version": "eiel_qrr_run_binding_v1",
                        "release_id": release_root.name,
                        "plan_sha256": "c" * 64,
                    }
                ),
                encoding="utf-8",
            )
        return {"ok": True, "command": command}

    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1._invoke_sealed_qrr_v1", _invoke
    )

    materialized = execute_science_request_v1(
        "materialize", request_path, qrr_root=tmp_path / "qrr", run_root=run_root
    )
    interrupted_receipt = {
        "schema_version": "eiel_qrr_run_receipt_v1",
        "release_id": release_root.name,
        "plan_sha256": "c" * 64,
        "status": "interrupted",
        "execution_receipt": {
            "ok": False,
            "plan_sha256": "c" * 64,
            "status": "interrupted",
        },
    }
    (run_root / "qrr_run_receipt.json").write_text(
        json.dumps(interrupted_receipt), encoding="utf-8"
    )
    continuation_core = {
        key: value for key, value in materialized.items() if key != "result_sha256"
    }
    continuation_core.update(
        {
            "ok": False,
            "command": "run",
            "qrr_result": {"ok": False, "run_receipt": interrupted_receipt},
        }
    )
    (run_root / "production_science_front_door_result.json").write_text(
        json.dumps(
            {
                **continuation_core,
                "result_sha256": canonical_sha256(continuation_core),
            }
        ),
        encoding="utf-8",
    )
    continued = execute_science_request_v1(
        "run", request_path, qrr_root=tmp_path / "qrr", run_root=run_root
    )

    assert materialized["ok"] is True
    assert continued["ok"] is True
    assert events == ["materialize", "preflight", "run"]


def test_science_front_door_recovers_exit_zero_result_from_exact_durable_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request_v2()), encoding="utf-8")
    release_root = tmp_path / "qrr/qrr1-active"
    release_root.mkdir(parents=True)
    run_root = tmp_path / "run"
    plan_sha256 = "c" * 64
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1.resolve_active_qrr_v2",
        lambda request, qrr_root: (
            release_root,
            _resolution_v2(request, release_root),
        ),
    )
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1.materialize_qrr_experiment_v2",
        lambda _request, release_root: {
            "schema_version": "eiel_qrr_experiment_v1",
            "deliberate_restore_ticks": [17, 33],
        },
    )

    def _invoke(argv):
        if argv[1] == "preflight":
            (run_root / "execution_plan.json").write_text(
                json.dumps(
                    {"plan_sha256": plan_sha256, "cells": [{"cell_id": "cell-a"}]}
                ),
                encoding="utf-8",
            )
            return {"ok": True, "preflight": _v2_preflight(plan_sha256)}
        execution = {
            "ok": True,
            "status": "completed",
            "plan_sha256": plan_sha256,
        }
        (run_root / "qrr_run_receipt.json").write_text(
            json.dumps(
                {
                    "schema_version": "eiel_qrr_run_receipt_v1",
                    "release_id": release_root.name,
                    "plan_sha256": plan_sha256,
                    "status": "completed",
                    "checkpoint_resume_exact": True,
                    "execution_receipt": execution,
                }
            ),
            encoding="utf-8",
        )
        (run_root / "qrr_run_binding.json").write_text(
            json.dumps(
                {
                    "schema_version": "eiel_qrr_run_binding_v1",
                    "release_id": release_root.name,
                    "plan_sha256": plan_sha256,
                    "deliberate_fresh_owner_restore_ticks": [17, 33],
                    "resume_invocations": 2,
                }
            ),
            encoding="utf-8",
        )
        (run_root / "campaign_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "eiel_campaign_execution_manifest_v1",
                    "status": "completed",
                    "counts": {
                        "completed": 1,
                        "failed": 0,
                        "interrupted": 0,
                        "pending": 0,
                        "running": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        raise _SealedQrrInvocationError(
            "stdout was polluted",
            returncode=0,
            stdout='diagnostic\n{"ok": true}\n',
            stderr="",
        )

    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1._invoke_sealed_qrr_v1", _invoke
    )
    monkeypatch.setattr(
        "engine.qrr.production_front_door_v1._start_canopy_relays_v1",
        lambda **_kwargs: {"state": "watching", "relays": []},
    )

    result = execute_science_request_v2(
        "run", request_path, qrr_root=tmp_path / "qrr", run_root=run_root
    )

    assert result["ok"] is True
    assert result["qrr_result"]["result_delivery"]["state"] == (
        "recovered_from_durable_completed_receipt"
    )
    assert not (run_root / "canopy_source_terminal.json").exists()


def _install_bound_v2_result_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    qrr_ok: bool,
) -> dict:
    import engine.qrr.production_front_door_v1 as qrr

    request = _request_v2()
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    qrr_root = tmp_path / "qrr"
    release_root = qrr_root / "qrr1-active"
    release_root.mkdir(parents=True)
    run_root = tmp_path / "run"
    plan_sha256 = "c" * 64
    experiment = {"schema_version": "eiel_qrr_experiment_v1"}
    release = SimpleNamespace(
        release_id=release_root.name,
        payload=lambda: {"manifest_sha256": "a" * 64},
    )
    envelope = SimpleNamespace(
        qualification_evidence_binding={"binding_sha256": "c" * 64}
    )
    events: list[str] = []

    monkeypatch.setattr(
        qrr,
        "resolve_active_qrr_v2",
        lambda normalized, qrr_root: (
            release_root,
            _resolution_v2(normalized, release_root),
        ),
    )
    monkeypatch.setattr(
        qrr,
        "materialize_qrr_experiment_v2",
        lambda _request, release_root: experiment,
    )
    monkeypatch.setattr(qrr, "load_qualified_runtime_release_v1", lambda _: release)
    monkeypatch.setattr(qrr, "_require_science_release_authority_v1", lambda _: envelope)

    def _invoke(argv):
        command = argv[1]
        events.append(command)
        if command == "preflight":
            plan = {
                "plan_sha256": plan_sha256,
                "execution": {
                    "acs_checkpoint_slots": 0,
                    "acs_capture_instrumentation": False,
                },
            }
            preflight = _v2_preflight(plan_sha256)
            (run_root / "execution_plan.json").write_text(
                json.dumps(plan), encoding="utf-8"
            )
            (run_root / "preflight_receipt.json").write_text(
                json.dumps(preflight), encoding="utf-8"
            )
            return {"ok": True, "command": command, "preflight": preflight}
        execution_receipt = {
            "ok": qrr_ok,
            "status": "completed" if qrr_ok else "failed",
            "plan_sha256": plan_sha256,
        }
        run_receipt = {
            "schema_version": "eiel_qrr_run_receipt_v1",
            "release_id": release_root.name,
            "plan_sha256": plan_sha256,
            "status": execution_receipt["status"],
            "execution_receipt": execution_receipt,
        }
        (run_root / "qrr_run_receipt.json").write_text(
            json.dumps(run_receipt), encoding="utf-8"
        )
        return {
            "ok": qrr_ok,
            "command": command,
            "status": execution_receipt["status"],
            "release_id": release_root.name,
            "run_root": str(run_root.resolve()),
            "run_receipt": run_receipt,
        }

    monkeypatch.setattr(qrr, "_invoke_sealed_qrr_v1", _invoke)
    monkeypatch.setattr(
        qrr,
        "_start_canopy_relays_v1",
        lambda **kwargs: qrr._record_canopy_observability_degraded_v1(
            kwargs["run_root"],
            plan=kwargs["plan"],
            reason="bounded fixture has no relay process",
        ),
    )
    return {
        "request_path": request_path,
        "qrr_root": qrr_root,
        "run_root": run_root,
        "release_root": release_root,
        "events": events,
    }


def test_retained_v2_result_recomputes_nested_outcome_and_baseline_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _install_bound_v2_result_fixture(
        tmp_path, monkeypatch, qrr_ok=False
    )
    first = execute_science_request_v2(
        "run",
        fixture["request_path"],
        qrr_root=fixture["qrr_root"],
        run_root=fixture["run_root"],
    )
    assert first["ok"] is False

    result_path = fixture["run_root"] / "production_science_front_door_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["ok"] = True
    result_core = dict(result)
    result_core.pop("result_sha256")
    result["result_sha256"] = canonical_sha256(result_core)
    result_path.write_text(json.dumps(result), encoding="utf-8")

    fixture["events"].clear()
    with pytest.raises(
        ProductionScienceFrontDoorError,
        match="retained result outcome mismatch",
    ):
        execute_science_request_v2(
            "run",
            fixture["request_path"],
            qrr_root=fixture["qrr_root"],
            run_root=fixture["run_root"],
        )
    assert fixture["events"] == []


@pytest.mark.parametrize(
    "mutation",
    (
        "release_id",
        "run_root",
        "resolution_receipt_sha256",
        "nested_release_id",
        "nested_run_root",
    ),
)
def test_retained_v2_result_identity_is_bound_to_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _install_bound_v2_result_fixture(tmp_path, monkeypatch, qrr_ok=True)
    first = execute_science_request_v2(
        "run",
        fixture["request_path"],
        qrr_root=fixture["qrr_root"],
        run_root=fixture["run_root"],
    )
    assert first["ok"] is True

    result_path = fixture["run_root"] / "production_science_front_door_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if mutation == "release_id":
        result["release_id"] = "other-release"
    elif mutation == "run_root":
        result["run_root"] = str(tmp_path / "other-run")
    elif mutation == "resolution_receipt_sha256":
        result["resolution_receipt_sha256"] = "f" * 64
    elif mutation == "nested_release_id":
        result["qrr_result"]["release_id"] = "other-release"
    else:
        result["qrr_result"]["run_root"] = str(tmp_path / "other-run")
    result_core = dict(result)
    result_core.pop("result_sha256")
    result["result_sha256"] = canonical_sha256(result_core)
    result_path.write_text(json.dumps(result), encoding="utf-8")

    fixture["events"].clear()
    with pytest.raises(
        ProductionScienceFrontDoorError,
        match="retained result identity mismatch",
    ):
        execute_science_request_v2(
            "run",
            fixture["request_path"],
            qrr_root=fixture["qrr_root"],
            run_root=fixture["run_root"],
        )
    assert fixture["events"] == []


@pytest.mark.parametrize("artifact", ("canopy", "qrr"))
@pytest.mark.parametrize("mutation", ("missing", "changed"))
def test_retained_v2_result_requires_exact_durable_run_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    mutation: str,
) -> None:
    fixture = _install_bound_v2_result_fixture(tmp_path, monkeypatch, qrr_ok=True)
    first = execute_science_request_v2(
        "run",
        fixture["request_path"],
        qrr_root=fixture["qrr_root"],
        run_root=fixture["run_root"],
    )
    assert first["ok"] is True

    filename = (
        "canopy_relay_supervision.json"
        if artifact == "canopy"
        else "qrr_run_receipt.json"
    )
    path = fixture["run_root"] / filename
    if mutation == "missing":
        path.unlink()
    else:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["status" if artifact == "qrr" else "state"] = "tampered"
        if artifact == "canopy":
            core = dict(receipt)
            core.pop("receipt_sha256")
            receipt["receipt_sha256"] = canonical_sha256(core)
        path.write_text(json.dumps(receipt), encoding="utf-8")

    fixture["events"].clear()
    with pytest.raises(
        (ProductionScienceFrontDoorError, ValueError),
        match="QRR|Canopy",
    ):
        execute_science_request_v2(
            "run",
            fixture["request_path"],
            qrr_root=fixture["qrr_root"],
            run_root=fixture["run_root"],
        )
    assert fixture["events"] == []


@pytest.mark.parametrize(
    "mutation",
    ("status", "manual_internal_decisions_required", "reasons"),
)
def test_bound_v2_resolution_requires_pass_without_manual_decisions_or_reasons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _install_bound_v2_result_fixture(tmp_path, monkeypatch, qrr_ok=True)
    first = execute_science_request_v2(
        "run",
        fixture["request_path"],
        qrr_root=fixture["qrr_root"],
        run_root=fixture["run_root"],
    )
    assert first["ok"] is True

    resolution_path = fixture["run_root"] / "active_qrr_resolution_receipt.json"
    resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
    if mutation == "status":
        resolution["status"] = "FAIL"
    elif mutation == "manual_internal_decisions_required":
        resolution["manual_internal_decisions_required"] = 1
    else:
        resolution["reasons"] = [{"code": "QRR_REPRESENTABILITY_BLOCKED"}]
    resolution_core = dict(resolution)
    resolution_core.pop("receipt_sha256")
    resolution["receipt_sha256"] = canonical_sha256(resolution_core)
    resolution_path.write_text(json.dumps(resolution), encoding="utf-8")

    request_receipt_path = fixture["run_root"] / "production_science_request_receipt.json"
    request_receipt = json.loads(request_receipt_path.read_text(encoding="utf-8"))
    request_receipt["resolution_receipt_sha256"] = resolution["receipt_sha256"]
    request_receipt["resolution_sha256"] = canonical_sha256(resolution)
    request_receipt_core = dict(request_receipt)
    request_receipt_core.pop("receipt_sha256")
    request_receipt["receipt_sha256"] = canonical_sha256(request_receipt_core)
    request_receipt_path.write_text(
        json.dumps(request_receipt), encoding="utf-8"
    )

    fixture["events"].clear()
    with pytest.raises(
        ProductionScienceFrontDoorError,
        match="bound active QRR resolution",
    ):
        execute_science_request_v2(
            "run",
            fixture["request_path"],
            qrr_root=fixture["qrr_root"],
            run_root=fixture["run_root"],
        )
    assert fixture["events"] == []
