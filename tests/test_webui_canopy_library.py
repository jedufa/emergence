from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen
import threading

import pytest

from infra.webui import canopy_library
from infra.webui import server as webui_server


def _write_relay(root: Path, *, source_run_id: str = "stored-run") -> Path:
    relay = root / "lineage-playback-relay"
    relay.mkdir(parents=True)
    playback = {"metadata": {"dataset_id": source_run_id}}
    provenance = {"schema_version": "eiel_lineage_playback_export_provenance_v1"}
    (relay / "playback_snapshot.json").write_text(json.dumps(playback), encoding="utf-8")
    (relay / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (relay / "relay_status.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    (relay / "relay_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": canopy_library.RELAY_SCHEMA,
                "sequence_number": 1,
                "source_run_id": source_run_id,
                "source_state": "complete",
                "semantics_mode": "actual_agents",
                "playback_file_name": "playback_snapshot.json",
                "playback_sha256": "a" * 64,
                "provenance_file_name": "provenance.json",
                "provenance_sha256": "b" * 64,
                "status_file_name": "relay_status.json",
            }
        ),
        encoding="utf-8",
    )
    return relay


def test_discovery_and_manifest_proxy_expose_only_declared_relay_artifacts(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    cold = tmp_path / "cold"
    _write_relay(runtime / "run-a", source_run_id="runtime-run")
    _write_relay(cold / "run-b", source_run_id="cold-run")
    roots = (("runtime", runtime), ("cold storage", cold))

    rows = canopy_library.discover(roots)

    by_label = {row["label"]: row for row in rows}
    assert set(by_label) == {"cold-run", "runtime-run"}
    assert by_label["cold-run"]["storage"] == "cold storage"
    assert "key=root0%2F" in by_label["runtime-run"]["manifest_url"]

    body, content_type = canopy_library.artifact(roots, key=by_label["runtime-run"]["key"], name="manifest")
    manifest = json.loads(body)
    assert content_type == "application/json; charset=utf-8"
    assert manifest["playback_file_name"].endswith("artifact=playback")
    assert "playback_snapshot.json" not in manifest["playback_file_name"]

    playback, _ = canopy_library.artifact(roots, key=by_label["runtime-run"]["key"], name="playback")
    assert json.loads(playback)["metadata"]["dataset_id"] == "runtime-run"
    with pytest.raises(canopy_library.CanopyLibraryError):
        canopy_library.artifact(roots, key=by_label["runtime-run"]["key"], name="secret")


def test_discovery_rejects_relay_artifacts_that_escape_the_relay_directory(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    relay = _write_relay(root / "run-a")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    manifest_path = relay / "relay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["playback_file_name"] = "../outside.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert canopy_library.discover((("runtime", root),)) == []


def test_webui_serves_canopy_at_same_origin_route_and_library_endpoint(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    _write_relay(runtime / "run-a")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), webui_server.EmergenceWebUIHandler)
    httpd.cfg = SimpleNamespace(canopy_library_roots=(("runtime", runtime),))  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        with urlopen(f"http://{host}:{port}/canopy", timeout=3) as response:
            page = response.read().decode("utf-8")
            assert response.geturl().endswith("/canopy/")
        with urlopen(f"http://{host}:{port}/api/canopy/playbacks", timeout=3) as response:
            library = json.load(response)
        assert "Living Lineage Canopy" in page
        assert "playbackLibrarySelect" in page
        assert library["playbacks"][0]["label"] == "stored-run"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)
