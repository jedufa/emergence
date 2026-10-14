from __future__ import annotations

from infra.webui import server


def test_default_chronicle_discovery_includes_runtime_repo_and_archive_locations() -> None:
    cfg = server._cfg_from_args(server.parse_args([]))

    assert (server.REPO_ROOT / "experiments" / "runs").resolve() in cfg.chronicle_roots
    assert (server.DEFAULT_RUNTIME_ROOT / "evidence_raw").resolve() in cfg.chronicle_roots
    assert (server.LEGACY_RUNTIME_ROOT / "evidence_raw").resolve() in cfg.chronicle_roots
    assert (server.REPO_ROOT / "chronicle_history").resolve() in cfg.chronicle_archive_roots
    assert (server.LEGACY_RUNTIME_ROOT / "chronicle_history").resolve() in cfg.chronicle_archive_roots
    assert server.COLD_CHRONICLE_ARCHIVE_ROOT.resolve() in cfg.chronicle_archive_roots


def test_chronicle_dialogue_exposes_human_run_search_and_locations() -> None:
    static_root = server.REPO_ROOT / "infra" / "webui" / "static"
    html = (static_root / "chronicle.html").read_text(encoding="utf-8")
    script = (static_root / "chronicle.js").read_text(encoding="utf-8")

    assert 'id="runSearch"' in html
    assert "matchingRuns" in script
    assert "run_path" in script
