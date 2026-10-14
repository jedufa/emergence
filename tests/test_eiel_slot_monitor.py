from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from engine.monitoring.slot_telemetry import (
    CPU_CAPACITY,
    GPU_CAPACITY,
    NullSlotLease,
    SlotLease,
    start_slot_lease,
)
from engine.campaigns.orchestration import BatchExecutionContextV1
from tools.slot_monitor.server import (
    SlotMonitorServer,
    _cpu_utilization_between,
    _generation_progress_metadata,
    _overlay_untracked_gpu_slots,
    build_snapshot,
    observatory_relay_manifest,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _lease(
    root: Path,
    *,
    device: str = "gpu",
    cells: int = 2,
    run_id: str = "campaign:batch-001",
    run_root: str = "/tmp/campaign-a",
    clock: FakeClock | None = None,
) -> SlotLease:
    cell_ids = tuple(f"cell-{index:03d}" for index in range(1, cells + 1))
    return SlotLease(
        device=device,
        run_id=run_id,
        cell_ids=cell_ids,
        current_generation=0,
        target_generation=100,
        ticks_per_generation=80,
        run_root=run_root,
        campaign_id="campaign",
        batch_id="batch-001",
        profile_id="test-profile",
        root=root,
        start_writer=False,
        clock=clock or FakeClock(),
    )


def _write_relay_fixture(run_root: Path) -> None:
    relay_root = run_root / "lineage-playback-relay"
    relay_root.mkdir(parents=True)
    playback = {"schema_version": "test_playback_v1", "agents": []}
    provenance = {
        "schema_version": "eiel_lineage_playback_export_provenance_v1",
        "output_sha256": "playback-sha",
    }
    status = {"state": "running", "playback_boundary": 7, "incomplete_artifact_warnings": ["partial"]}
    cross_check = {"ok": True}
    (relay_root / "playback_snapshot.json").write_text(json.dumps(playback), encoding="utf-8")
    (relay_root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (relay_root / "relay_status.json").write_text(json.dumps(status), encoding="utf-8")
    (relay_root / "cross_check.json").write_text(json.dumps(cross_check), encoding="utf-8")
    (relay_root / "relay_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "llc_near_live_playback_relay_v0_manifest",
                "sequence_number": 3,
                "source_run_id": "run-with-canopy",
                "source_state": "running",
                "latest_exported_playback_boundary": 6,
                "semantics_mode": "actual_agents",
                "playback_file_name": "playback_snapshot.json",
                "playback_sha256": "playback-sha",
                "provenance_file_name": "provenance.json",
                "provenance_sha256": "provenance-sha",
                "status_file_name": "relay_status.json",
                "cross_check_file_name": "cross_check.json",
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )


def test_slot_lease_publishes_monotonic_tick_speed_and_eta(tmp_path: Path) -> None:
    clock = FakeClock()
    lease = _lease(tmp_path, clock=clock)
    clock.advance(20)
    lease.update(current_generation=4, status="running", stage="checkpoint G4")

    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "eiel_simulation_slot_heartbeat_v2"
    assert payload["device"] == "gpu"
    assert payload["next_absolute_tick"] == 320
    assert payload["rate_ticks_per_second"] == pytest.approx(16.0)
    assert payload["eta_seconds"] == pytest.approx(480.0)
    assert [row["position"] for row in payload["cells"]] == [1, 2]
    assert payload["cells"][0]["progress_fraction"] == pytest.approx(0.04)

    with pytest.raises(ValueError, match="monotonic"):
        lease.update(current_generation=3)
    lease.close(status="complete")
    terminal = json.loads(lease.path.read_text(encoding="utf-8"))
    assert terminal["status"] == "complete"
    assert terminal["next_absolute_tick"] == 8_000


def test_slot_lease_accepts_authoritative_in_generation_tick_progress(tmp_path: Path) -> None:
    clock = FakeClock()
    lease = _lease(tmp_path, clock=clock)
    clock.advance(2)
    lease.update(next_absolute_tick=13, status="running", stage="tick 13")

    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    assert payload["next_absolute_tick"] == 13
    assert payload["target_absolute_tick"] == 8_000
    assert payload["rate_ticks_per_second"] == pytest.approx(6.5)
    assert payload["cells"][0]["progress_fraction"] == pytest.approx(13 / 8_000)

    with pytest.raises(ValueError, match="monotonic"):
        lease.update(next_absolute_tick=12)
    lease.close(status="interrupted")


def test_slot_lease_accepts_direct_tick_native_bounds(tmp_path: Path) -> None:
    clock = FakeClock()
    lease = SlotLease(
        device="cpu",
        run_id="tick-native",
        cell_ids=("tick-native",),
        start_absolute_tick=10,
        target_absolute_tick=17,
        run_root=tmp_path,
        root=tmp_path,
        start_writer=False,
        clock=clock,
    )
    clock.advance(2)
    lease.update(next_absolute_tick=14, status="running", stage="tick partition committed at 14")

    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    assert payload["start_absolute_tick"] == 10
    assert payload["next_absolute_tick"] == 14
    assert payload["target_absolute_tick"] == 17
    assert payload["rate_ticks_per_second"] == pytest.approx(2.0)
    lease.close(status="complete")


def test_slot_lease_projects_shared_gpu_owner_and_scheduler_fence(
    tmp_path: Path,
) -> None:
    lease = SlotLease(
        device="gpu",
        run_id="shared-route:batch-0001",
        cell_ids=("physical-batch:batch-0001",),
        start_absolute_tick=0,
        target_absolute_tick=8,
        run_root=tmp_path,
        logical_world_count=128,
        gpu_admission_mode="shared_functional",
        exclusive_gpu=False,
        peak_vram_gib=1.25,
        workload_class="cuda_full_evolution_v2_shared_functional",
        gpu_device_id="GPU-TEST-UUID",
        scheduler_resource_lease_id="LPRSTART-TEST",
        scheduler_resource_ids=(
            "gpu:GPU-TEST-UUID/lane/cuda-full-evolution-v2-w32-shared-v1/0",
        ),
        scheduler_fencing_token_digest="a" * 64,
        root=tmp_path,
        start_writer=False,
        clock=FakeClock(),
    )

    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    assert payload["logical_world_count"] == 128
    assert payload["gpu_admission_mode"] == "shared_functional"
    assert payload["exclusive_gpu"] is False
    assert payload["peak_vram_gib"] == 1.25
    assert payload["gpu_device_id"] == "GPU-TEST-UUID"
    assert payload["scheduler_resource_lease_id"] == "LPRSTART-TEST"
    assert payload["scheduler_fencing_token_digest"] == "a" * 64


def test_shared_gpu_slot_requires_declared_peak(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires peak_vram_gib"):
        SlotLease(
            device="gpu",
            run_id="shared-route",
            cell_ids=("physical-batch",),
            start_absolute_tick=0,
            target_absolute_tick=1,
            run_root=tmp_path,
            gpu_admission_mode="shared_functional",
            exclusive_gpu=False,
            root=tmp_path,
            start_writer=False,
        )


def test_telemetry_setup_failure_is_fail_open(tmp_path: Path) -> None:
    unusable_root = tmp_path / "not-a-directory"
    unusable_root.write_text("occupied", encoding="utf-8")
    lease = start_slot_lease(
        device="gpu",
        run_id="fail-open",
        cell_ids=("cell-1",),
        current_generation=0,
        target_generation=2,
        ticks_per_generation=80,
        run_root=tmp_path,
        root=unusable_root,
    )
    assert isinstance(lease, NullSlotLease)
    lease.update(current_generation=1)
    lease.close(status="failed")


@pytest.mark.parametrize("cells", [1, 2, GPU_CAPACITY])
def test_snapshot_maps_physical_gpu_positions_and_keeps_rest_idle(
    tmp_path: Path, cells: int
) -> None:
    lease = _lease(tmp_path, cells=cells)
    lease.update(current_generation=5, status="running", stage="native batch")

    snapshot = build_snapshot(tmp_path, include_system=False)
    assert len(snapshot["gpu_slots"]) == GPU_CAPACITY
    assert len(snapshot["cpu_slots"]) == CPU_CAPACITY
    assert snapshot["gpu_active"] == cells
    assert [row["position"] for row in snapshot["gpu_slots"][:cells]] == list(
        range(1, cells + 1)
    )
    assert all(row["status"] == "running" for row in snapshot["gpu_slots"][:cells])
    assert all(row["status"] == "idle" for row in snapshot["gpu_slots"][cells:])
    lease.close(status="interrupted")


def test_cpu_contract_is_exactly_one_position(tmp_path: Path) -> None:
    lease = _lease(tmp_path, device="cpu", cells=1)
    lease.update(current_generation=1)
    snapshot = build_snapshot(tmp_path, include_system=False)
    assert snapshot["cpu_active"] == 1
    assert snapshot["cpu_slots"][0]["label"] == "CPU 01"
    assert snapshot["cpu_slots"][0]["cell_id"] == "cell-001"
    with pytest.raises(ValueError, match="exactly one"):
        _lease(tmp_path, device="cpu", cells=2, run_root="/tmp/cpu-b")
    lease.close()


def test_cpu_utilization_uses_aggregate_counter_delta() -> None:
    assert _cpu_utilization_between((400, 1000), (450, 1200)) == pytest.approx(75.0)
    assert _cpu_utilization_between((450, 1200), (450, 1200)) is None


def test_untracked_gpu_process_occupies_inferred_slots() -> None:
    slots = [
        {
            "device": "gpu",
            "position": index,
            "label": f"GPU {index:02d}",
            "status": "idle",
        }
        for index in range(1, GPU_CAPACITY + 1)
    ]
    _overlay_untracked_gpu_slots(
        slots,
        [
            {
                "pid": 123,
                "process": "python3",
                "run_id": "venv1-p1-b2g100-calibrated",
                "run_root": "/tmp/run",
                "inferred_slot_count": 2,
                "cell_ids": ["cell-gd", "cell-fi"],
                "target_absolute_tick": 8_000,
                "tick_progress_available": True,
                "status": "running",
                "next_absolute_tick": 1_040,
                "progress_fraction": 0.13,
                "rate_ticks_per_second": 2.0,
                "eta_seconds": 3480.0,
                "stage": "checkpoint-chain-g0-g25 · G13 · inter_tick_control",
            }
        ],
    )
    assert [row["status"] for row in slots[:3]] == ["running", "running", "idle"]
    assert slots[0]["cell_id"] == "cell-gd"
    assert slots[1]["cell_id"] == "cell-fi"
    assert slots[0]["next_absolute_tick"] == 1_040
    assert slots[1]["target_absolute_tick"] == 8_000


def test_generation_progress_projects_tick_rate_and_eta(tmp_path: Path) -> None:
    route = tmp_path / "attempt-0001" / "route-c"
    owner = route / "owner"
    owner.mkdir(parents=True)
    metrics = owner / "metrics.csv"
    metrics.write_text("generation\n", encoding="utf-8")
    progress = route / "generation_progress.json"
    progress.write_text(
        json.dumps(
            {
                "schema_version": "eiel_cuda_architecture_b_generation_progress_v1",
                "active_generation": 4,
                "completed_generation_boundary": 4,
                "failure": None,
                "phase": "running",
                "route_id": "checkpoint-chain-g0-g25",
                "run_control_status": "inter_tick_control",
                "start_generation": 0,
                "target_next_generation": 25,
            }
        ),
        encoding="utf-8",
    )
    metrics.touch()
    progress.touch()
    metrics_mtime = metrics.stat().st_mtime
    progress_mtime = metrics_mtime + 100.0
    os.utime(progress, (progress_mtime, progress_mtime))
    observed = _generation_progress_metadata(
        tmp_path, target_generation=10, ticks_per_generation=80
    )
    assert observed["next_absolute_tick"] == 320
    assert observed["target_absolute_tick"] == 800
    assert observed["progress_fraction"] == pytest.approx(0.4)
    assert observed["rate_ticks_per_second"] == pytest.approx(3.2)
    assert observed["eta_seconds"] == pytest.approx(150.0)


def test_conflicting_allocations_are_visible(tmp_path: Path) -> None:
    first = _lease(tmp_path, run_id="first", run_root="/tmp/first")
    second = _lease(tmp_path, run_id="second", run_root="/tmp/second")
    third = _lease(tmp_path, run_id="third", run_root="/tmp/third")
    first.update(current_generation=1)
    second.update(current_generation=2)
    third.update(current_generation=3)
    snapshot = build_snapshot(tmp_path, include_system=False)
    assert snapshot["gpu_slots"][0]["status"] == "conflict"
    assert len(snapshot["gpu_slots"][0]["conflicts"]) == 2
    assert {
        snapshot["gpu_slots"][0]["run_id"],
        *(row["run_id"] for row in snapshot["gpu_slots"][0]["conflicts"]),
    } == {"first", "second", "third"}
    first.close(status="interrupted")
    second.close(status="interrupted")
    third.close(status="interrupted")


def test_retained_terminal_allocation_does_not_conflict_with_new_owner(
    tmp_path: Path,
) -> None:
    previous = _lease(tmp_path, run_id="previous", run_root="/tmp/previous")
    previous.close(status="complete")
    current = _lease(tmp_path, run_id="current", run_root="/tmp/current")
    current.update(current_generation=2, status="running")

    snapshot = build_snapshot(tmp_path, include_system=False)
    assert snapshot["gpu_slots"][0]["status"] == "running"
    assert snapshot["gpu_slots"][0]["run_id"] == "current"
    assert snapshot["gpu_active"] == 2
    current.close(status="interrupted")


def test_retained_terminal_allocations_do_not_conflict_with_each_other(
    tmp_path: Path,
) -> None:
    first = _lease(tmp_path, run_id="first", run_root="/tmp/first")
    first.close(status="complete")
    second = _lease(tmp_path, run_id="second", run_root="/tmp/second")
    second.close(status="complete")

    snapshot = build_snapshot(tmp_path, include_system=False)

    assert snapshot["gpu_slots"][0]["status"] == "complete"
    assert snapshot["gpu_slots"][0]["conflicts"] == []


def test_stale_allocations_do_not_conflict_with_each_other(tmp_path: Path) -> None:
    first = _lease(tmp_path, run_id="first", run_root="/tmp/first")
    second = _lease(tmp_path, run_id="second", run_root="/tmp/second")
    stale_now = max(first.path.stat().st_mtime, second.path.stat().st_mtime) + 16.0

    snapshot = build_snapshot(tmp_path, include_system=False, now=stale_now)

    assert snapshot["gpu_slots"][0]["status"] == "stale"
    assert snapshot["gpu_slots"][0]["conflicts"] == []


def test_dead_producer_releases_slots_immediately(tmp_path: Path) -> None:
    lease = _lease(tmp_path)
    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    payload["pid"] = 999_999_999
    lease.path.write_text(json.dumps(payload), encoding="utf-8")
    snapshot = build_snapshot(tmp_path, include_system=False)
    assert snapshot["gpu_active"] == 0
    assert all(row["status"] == "idle" for row in snapshot["gpu_slots"])


def test_implausible_tick_rate_is_filtered_from_snapshot(tmp_path: Path) -> None:
    lease = _lease(tmp_path)
    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    payload["cells"][0]["rate_ticks_per_second"] = 1_000_000_001.0
    lease.path.write_text(json.dumps(payload), encoding="utf-8")

    snapshot = build_snapshot(tmp_path, include_system=False)

    assert snapshot["gpu_slots"][0]["rate_ticks_per_second"] is None


def test_legacy_heartbeat_projects_only_its_committed_tick_boundary(
    tmp_path: Path,
) -> None:
    lease = _lease(tmp_path)
    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    payload["schema_version"] = "eiel_simulation_slot_heartbeat_v1"
    payload["ticks_per_generation"] = 80
    payload["cells"][0].pop("next_absolute_tick")
    payload["cells"][0].pop("target_absolute_tick")
    payload["cells"][0].pop("rate_ticks_per_second")
    payload["cells"][0].update(
        current_generation=0,
        target_generation=100,
        rate_generations_per_second=None,
    )
    lease.path.write_text(json.dumps(payload), encoding="utf-8")

    snapshot = build_snapshot(tmp_path, include_system=False)

    assert snapshot["gpu_slots"][0]["status"] == "starting"
    assert snapshot["gpu_slots"][0]["next_absolute_tick"] == 0
    assert snapshot["gpu_slots"][0]["target_absolute_tick"] == 8_000
    assert snapshot["gpu_slots"][0]["tick_precision"] == "boundary-only"


def test_live_simulations_ui_does_not_render_generation_metrics() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "infra"
        / "webui"
        / "static"
        / "live_simulations.js"
    ).read_text(encoding="utf-8").lower()

    assert "generation" not in source
    assert "gen/h" not in source
    assert "rate_ticks_per_second" in source


def test_expired_stale_allocation_releases_slots(tmp_path: Path) -> None:
    lease = _lease(tmp_path)
    heartbeat_time = lease.path.stat().st_mtime

    snapshot = build_snapshot(
        tmp_path,
        include_system=False,
        now=heartbeat_time + 301.0,
    )

    assert snapshot["gpu_active"] == 0
    assert all(row["status"] == "idle" for row in snapshot["gpu_slots"])


def test_campaign_checkpoint_callback_advances_operational_heartbeat(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry"
    lease = _lease(telemetry)
    manifest_path = tmp_path / "batch" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"status": "running", "checkpoint": None}), encoding="utf-8"
    )
    context = BatchExecutionContextV1(
        run_root=tmp_path,
        batch_root=manifest_path.parent,
        manifest_path=manifest_path,
        slot_lease=lease,
    )

    context.record_checkpoint({"generation": 7, "path": "checkpoint.json"})

    heartbeat = json.loads(lease.path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert heartbeat["next_absolute_tick"] == 560
    assert heartbeat["stage"] == "checkpoint tick 560"
    assert manifest["checkpoint"]["generation"] == 7
    lease.close(status="interrupted")


def test_batch_context_publishes_native_dispatch_progress_between_checkpoints(
    tmp_path: Path,
) -> None:
    lease = SlotLease(
        device="gpu",
        run_id="native-run",
        cell_ids=("native-run",),
        start_absolute_tick=0,
        target_absolute_tick=1_000,
        run_root=tmp_path,
        root=tmp_path / "telemetry",
        start_writer=False,
    )
    context = BatchExecutionContextV1(
        run_root=tmp_path,
        batch_root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        slot_lease=lease,
    )

    context.record_progress(640)

    heartbeat = json.loads(lease.path.read_text(encoding="utf-8"))
    assert heartbeat["next_absolute_tick"] == 640
    assert heartbeat["cells"][0]["next_absolute_tick"] == 640
    assert heartbeat["stage"] == "native dispatch tick 640"
    lease.close(status="interrupted")


def test_read_only_http_surface_serves_health_snapshot_and_page(tmp_path: Path) -> None:
    lease = _lease(tmp_path)
    lease.update(current_generation=2)
    server = SlotMonitorServer(("127.0.0.1", 0), tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        with urlopen(f"http://{host}:{port}/api/v1/health", timeout=3) as response:
            health = json.load(response)
        with urlopen(f"http://{host}:{port}/api/v1/snapshot", timeout=3) as response:
            snapshot = json.load(response)
        with urlopen(f"http://{host}:{port}/", timeout=3) as response:
            page = response.read().decode("utf-8")
        with urlopen(f"http://{host}:{port}/living-lineage-canopy/index.html", timeout=3) as response:
            canopy_page = response.read().decode("utf-8")
        assert health["ok"] is True
        assert len(snapshot["gpu_slots"]) == 72
        assert "Simulation Slots" in page
        assert "72 resident GPU worlds" in page
        assert "Living Lineage Canopy" in canopy_page
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        lease.close(status="interrupted")


def test_observatory_snapshot_marks_slots_with_lineage_canopy(tmp_path: Path) -> None:
    run_root = tmp_path / "run-with-canopy"
    _write_relay_fixture(run_root)
    lease = _lease(tmp_path / "telemetry", cells=1, run_id="run-with-canopy", run_root=str(run_root))
    try:
        snapshot = build_snapshot(tmp_path / "telemetry", include_system=False)
    finally:
        lease.close(status="interrupted")

    observatory = snapshot["gpu_slots"][0]["observatory"]
    assert observatory["canopy_available"] is True
    assert observatory["canopy_url"].startswith("/living-lineage-canopy/index.html?relay=")
    assert observatory["semantics"] == "actual_agents"
    assert observatory["relay_sequence"] == 3
    assert observatory["relay_boundary"] == 7
    assert observatory["relay_warning_count"] == 1
    assert snapshot["gpu_slots"][1]["observatory"]["canopy_available"] is False


def test_observatory_exposes_tick_native_canopy_boundary_without_enabling_link(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "tick-native-run"
    run_root.mkdir()
    (run_root / "tick_native_observability.json").write_text(
        json.dumps(
            {
                "schema_version": "eiel_tick_native_observability_contract_v1",
                "route_identity": "tick_native_execution_route_v1",
                "live_simulations": {
                    "runtime_partition_ticks": 7,
                    "telemetry_precision": "committed_partition_cursor",
                },
                "canopy": {
                    "relay_compatible": False,
                    "checkpoint_every_ticks": 10,
                    "reason": "reference checkpoints have no Canopy lineage projection",
                },
            }
        ),
        encoding="utf-8",
    )
    lease = _lease(tmp_path / "telemetry", cells=1, run_id="tick-native", run_root=str(run_root))
    try:
        snapshot = build_snapshot(tmp_path / "telemetry", include_system=False)
    finally:
        lease.close(status="interrupted")

    observatory = snapshot["gpu_slots"][0]["observatory"]
    assert observatory["canopy_available"] is False
    assert observatory["canopy_url"] is None
    assert observatory["canopy_checkpoint_every_ticks"] == 10
    assert observatory["canopy_reason"] == "reference checkpoints have no Canopy lineage projection"
    assert observatory["tick_native"] == {
        "runtime_partition_ticks": 7,
        "telemetry_precision": "committed_partition_cursor",
        "canopy_relay_compatible": False,
        "canopy_checkpoint_every_ticks": 10,
        "canopy_reason": "reference checkpoints have no Canopy lineage projection",
    }


def test_observatory_proxy_rewrites_manifest_without_exposing_filesystem_path(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-with-canopy"
    _write_relay_fixture(run_root)
    lease = _lease(tmp_path / "telemetry", cells=1, run_id="run-with-canopy", run_root=str(run_root))
    try:
        manifest = observatory_relay_manifest(
            tmp_path / "telemetry",
            device="gpu",
            position=1,
        )
    finally:
        lease.close(status="interrupted")

    assert manifest["playback_sha256"] == "playback-sha"
    assert manifest["playback_file_name"] == "/api/v1/observatory/relay/gpu/1/file/playback_snapshot.json"
    assert manifest["provenance_file_name"] == "/api/v1/observatory/relay/gpu/1/file/provenance.json"
    assert manifest["status_file_name"] == "/api/v1/observatory/relay/gpu/1/file/relay_status.json"
    assert str(tmp_path) not in json.dumps(manifest)


def test_observatory_http_proxy_serves_only_manifest_listed_files(tmp_path: Path) -> None:
    run_root = tmp_path / "run-with-canopy"
    _write_relay_fixture(run_root)
    (run_root / "lineage-playback-relay" / "secret.json").write_text("{}", encoding="utf-8")
    lease = _lease(tmp_path / "telemetry", cells=1, run_id="run-with-canopy", run_root=str(run_root))
    server = SlotMonitorServer(("127.0.0.1", 0), tmp_path / "telemetry")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        with urlopen(f"http://{host}:{port}/api/v1/observatory/relay/gpu/1/manifest", timeout=3) as response:
            manifest = json.load(response)
        with urlopen(f"http://{host}:{port}{manifest['playback_file_name']}", timeout=3) as response:
            playback = json.load(response)
        assert playback["schema_version"] == "test_playback_v1"
        with pytest.raises(HTTPError) as excinfo:
            urlopen(f"http://{host}:{port}/api/v1/observatory/relay/gpu/1/file/secret.json", timeout=3)
        assert excinfo.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        lease.close(status="interrupted")


def test_static_page_is_standalone_and_has_no_control_actions() -> None:
    static = Path(__file__).parents[1] / "tools" / "slot_monitor" / "static"
    combined = "\n".join(
        (static / name).read_text(encoding="utf-8")
        for name in ("index.html", "app.js", "style.css")
    )
    assert "/api/v1/snapshot" in combined
    assert "orderedSlots(slots).map(slotMarkup)" in combined
    assert 'new Set(["starting", "running", "finishing", "stale", "conflict", "untracked"])' in combined
    assert "const REFRESH_INTERVAL_MS = 2000" in combined
    assert combined.index('id="cpuGrid"') < combined.index('id="gpuGrid"')
    assert "#matrix" not in combined
    assert "stop" not in combined.lower()
    assert "delete" not in combined.lower()
