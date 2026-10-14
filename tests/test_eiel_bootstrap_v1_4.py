from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tests.eiel_actor_v14_helpers import FIXED_V14, make_bootstrap_repo, run_bootstrap
from tests.eiel_actor_v11_helpers import run_eiel
from tools.actors.bootstrap import bootstrap_refresh_arguments


def test_bootstrap_passes_realistic_git_repo_and_retains_receipt(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    result = run_bootstrap(repo=repo, state_root=state)
    assert result["status"] == "READY_WITH_WARNINGS"
    bootstrap = result["bootstrap"]
    assert bootstrap["bootstrap_id"].startswith("BST-")
    assert bootstrap["repository"]["root"] == str(repo.resolve())
    assert bootstrap["environment"]["source"] == "repository_venv"
    assert bootstrap["authority"]["preflight"]["catalog_build_id"]
    assert not bootstrap["blocked_check_names"]
    assert any(
        row["name"] == "operator_instruction" and row["status"] == "WARN"
        for row in bootstrap["checks"]
    )
    stored = state / "bootstrap/records" / f"{bootstrap['bootstrap_id']}.json"
    assert stored.is_file()


def test_bootstrap_preserves_selected_venv_symlink_path_for_runtime_probe(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    selected_python = repo / ".venv/bin/python"

    result = run_bootstrap(
        repo=repo,
        state_root=state,
        extra=("--python", str(selected_python)),
    )

    environment = result["bootstrap"]["environment"]
    assert environment["python"] == str(selected_python)
    assert environment["resolved_python"] == str(selected_python.resolve())
    assert environment["python_file"]["path"] == str(selected_python)
    assert environment["python_file"]["resolved_path"] == str(selected_python.resolve())


def test_bootstrap_rejects_extracted_non_git_repository(tmp_path: Path) -> None:
    repo = tmp_path / "extracted"
    repo.mkdir()
    state = tmp_path / "state"
    result = run_eiel(
        [
            "bootstrap", "inspect",
            "--repo-root", str(repo),
            "--task-mode", "mechanical",
            "--operator-instruction-ref", "test",
            "--preflight-exemption-reason", "non-substantive packaging check",
            "--state-root", str(state),
            "--generated-at", FIXED_V14,
        ],
        state_root=state,
        expected=2,
    )
    assert result["status"] == "BLOCKED"
    assert "git_worktree" in result["bootstrap"]["blocked_check_names"]


def test_bootstrap_rejects_dirty_editing_repo_without_acknowledgement(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    (repo / "engine/dirty.py").write_text("x=1\n", encoding="utf-8")
    state = tmp_path / "state"
    result = run_bootstrap(repo=repo, state_root=state, expected=2)
    assert result["status"] == "BLOCKED"
    assert "dirty_state" in result["bootstrap"]["blocked_check_names"]
    accepted = run_bootstrap(repo=repo, state_root=state, acknowledge_dirty=True, generated_at="2026-08-04T00:01:00Z")
    assert accepted["status"] == "READY_WITH_WARNINGS"


def test_bootstrap_rejects_gpu_work_when_cuda_unavailable(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    result = run_bootstrap(repo=repo, state_root=state, expected=2, gpu_required="yes")
    assert result["status"] == "BLOCKED"
    assert "cuda" in result["bootstrap"]["blocked_check_names"]


def test_bootstrap_rejects_preflight_bound_to_different_catalog(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    receipt = state / "bad-preflight.json"
    good = run_bootstrap(repo=repo, state_root=state)
    retained = Path(good["bootstrap"]["authority"]["preflight_receipt_path"])
    payload = json.loads(retained.read_text(encoding="utf-8"))
    payload["source_digest"] = "0" * 64
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    result = run_eiel(
        [
            "bootstrap", "inspect",
            "--repo-root", str(repo),
            "--task-mode", "implementation",
            "--topic", "bad catalog binding",
            "--operator-instruction-ref", "test",
            "--edit-authorized",
            "--write-scope", "tools",
            "--expected-validation", "pytest -q",
            "--preflight-receipt", str(receipt),
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:02:00Z",
        ],
        state_root=state,
        expected=2,
    )
    assert result["status"] == "BLOCKED"
    assert "librarian_preflight" in result["bootstrap"]["blocked_check_names"]


def test_bootstrap_retains_explicit_bounded_fallback_when_preflight_allows_it(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    good = run_bootstrap(repo=repo, state_root=state)
    retained = Path(good["bootstrap"]["authority"]["preflight_receipt_path"])
    payload = json.loads(retained.read_text(encoding="utf-8"))
    payload["fail_closed"] = True
    payload["fallback_repository_search_allowed"] = True
    payload["authority_assertion_ids"] = []
    payload["authority_domains"] = []
    payload["unresolved_authority_gaps"] = ["No explicit assertion matched the bounded topic."]
    receipt = state / "fallback-preflight.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    result = run_eiel(
        [
            "bootstrap", "inspect",
            "--repo-root", str(repo),
            "--task-mode", "implementation",
            "--topic", "bounded repository fallback",
            "--operator-instruction-ref", "test",
            "--edit-authorized",
            "--write-scope", "tools",
            "--expected-validation", "pytest -q",
            "--preflight-receipt", str(receipt),
            "--preflight-exemption-reason", "operator prompt governs; no broader authority claim",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:03:00Z",
        ],
        state_root=state,
    )

    assert result["status"] == "READY_WITH_WARNINGS"
    check = next(row for row in result["bootstrap"]["checks"] if row["name"] == "librarian_preflight")
    assert check["status"] == "WARN"
    assert check["details"]["fallback_repository_search_allowed"] is True
    assert check["details"]["fallback_reason"] == "operator prompt governs; no broader authority claim"

    refreshed = run_eiel(
        [
            "bootstrap", "refresh", result["bootstrap"]["bootstrap_id"],
            "--repo-root", str(repo),
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:04:00Z",
        ],
        state_root=state,
    )
    assert refreshed["status"] == "READY_WITH_WARNINGS"
    assert refreshed["bootstrap"]["refresh_inputs"]["preflight_exemption_reason"] == (
        "operator prompt governs; no broader authority claim"
    )


def test_bootstrap_is_deterministic_with_fixed_inputs(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state_a = tmp_path / "state-a"
    state_b = tmp_path / "state-b"
    first = run_bootstrap(repo=repo, state_root=state_a)
    second = run_bootstrap(repo=repo, state_root=state_b)
    assert first["bootstrap"]["bootstrap_id"] == second["bootstrap"]["bootstrap_id"]
    clean_a = dict(first["bootstrap"])
    clean_b = dict(second["bootstrap"])
    clean_a["authority"]["preflight_receipt_path"] = "<state>"
    clean_b["authority"]["preflight_receipt_path"] = "<state>"
    clean_a["authority"]["preflight"]["receipt_path"] = "<state>"
    clean_b["authority"]["preflight"]["receipt_path"] = "<state>"
    assert clean_a == clean_b


def test_bootstrap_default_status_reports_latest_receipt(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    created = run_bootstrap(repo=repo, state_root=state)
    latest = run_eiel(["bootstrap", "status", "--state-root", str(state)], state_root=state)
    assert latest["record"]["record_id"] == created["bootstrap"]["bootstrap_id"]


def test_bootstrap_refresh_rebinds_live_preflight_after_head_change(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    evidence_root = tmp_path / "external-evidence"
    evidence_root.mkdir()
    original = run_bootstrap(
        repo=repo,
        state_root=state,
        extra=("--external-evidence-root", str(evidence_root)),
    )
    changed = repo / "tools" / "post_bootstrap_change.py"
    changed.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", str(changed)], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "advance fixture head"], check=True)

    previous_record = json.loads(
        (state / "bootstrap/records" / f"{original['bootstrap']['bootstrap_id']}.json")
        .read_text(encoding="utf-8")
    )
    refresh_arguments = bootstrap_refresh_arguments(
        previous_record,
        repo_root=repo,
        state_root=state,
    )
    assert "--preflight-receipt" not in refresh_arguments

    refreshed = run_eiel(
        [
            "bootstrap", "refresh", original["bootstrap"]["bootstrap_id"],
            "--repo-root", str(repo),
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:01:00Z",
        ],
        state_root=state,
        expected=2,
    )

    assert refreshed["status"] == "BLOCKED"
    assert refreshed["refreshed_from"] == original["bootstrap"]["bootstrap_id"]
    assert refreshed["bootstrap"]["repository"]["head"] != original["bootstrap"]["repository"]["head"]
    assert refreshed["bootstrap"]["task"]["operator_instruction_ref"] == "test operator instruction"
    assert refreshed["bootstrap"]["task"]["external_evidence_roots"] == [
        {"path": str(evidence_root.resolve()), "exists": True, "kind": "directory"}
    ]


def test_bootstrap_records_external_evidence_root_without_expanding_write_scope(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    evidence_root = tmp_path / "external-evidence"
    evidence_root.mkdir()

    result = run_bootstrap(
        repo=repo,
        state_root=state,
        extra=("--external-evidence-root", str(evidence_root)),
    )

    bootstrap = result["bootstrap"]
    assert bootstrap["task"]["write_scope"] == [{"path": "tools", "exists": True, "kind": "directory"}]
    assert bootstrap["task"]["external_evidence_roots"] == [
        {"path": str(evidence_root.resolve()), "exists": True, "kind": "directory"}
    ]
    assert any(
        row["name"] == "external_evidence_root" and row["status"] == "PASS"
        for row in bootstrap["checks"]
    )


def test_bootstrap_still_rejects_external_write_scope_when_evidence_root_is_declared(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    evidence_root = tmp_path / "external-evidence"
    evidence_root.mkdir()

    result = run_bootstrap(
        repo=repo,
        state_root=state,
        expected=2,
        write_scope=(str(evidence_root),),
        extra=("--external-evidence-root", str(evidence_root)),
    )

    assert result["status"] == "BLOCKED"
    assert "write_scope" in result["bootstrap"]["blocked_check_names"]
    assert result["bootstrap"]["task"]["external_evidence_roots"] == [
        {"path": str(evidence_root.resolve()), "exists": True, "kind": "directory"}
    ]


def test_bootstrap_rejects_repository_path_as_external_evidence_root(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"

    result = run_bootstrap(
        repo=repo,
        state_root=state,
        expected=2,
        extra=("--external-evidence-root", str(repo / "evidences")),
    )

    assert result["status"] == "BLOCKED"
    assert "external_evidence_root" in result["bootstrap"]["blocked_check_names"]


def test_bootstrap_refresh_rejects_another_repository_lane(tmp_path: Path) -> None:
    first = make_bootstrap_repo(tmp_path / "first")
    second = make_bootstrap_repo(tmp_path / "second")
    state = tmp_path / "state"
    original = run_bootstrap(repo=first, state_root=state)

    result = run_eiel(
        [
            "bootstrap", "refresh", original["bootstrap"]["bootstrap_id"],
            "--repo-root", str(second),
            "--state-root", str(state),
        ],
        state_root=state,
        expected=2,
    )

    assert result["reason_codes"] == ["bootstrap_refresh_repository_mismatch"]


def test_bootstrap_accepts_authority_domain_and_stores_filter(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    result = run_eiel(
        [
            "bootstrap", "inspect",
            "--repo-root", str(repo),
            "--task-mode", "implementation",
            "--topic", "qualification authority register",
            "--operator-instruction-ref", "test",
            "--edit-authorized",
            "--write-scope", "tools",
            "--expected-validation", "pytest -q",
            "--authority-domain", "execution_route",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:10:00Z",
        ],
        state_root=state,
    )
    assert result["status"] in {"READY", "READY_WITH_WARNINGS"}
    bootstrap = result["bootstrap"]
    assert bootstrap["task"]["authority_domain"] == "execution_route"
    assert bootstrap["authority"]["preflight"]["authority_domain_filter"] == "execution_route"
    assert bootstrap["authority"]["preflight"]["authority_resolution_state"] == "resolved"


def test_bootstrap_ambiguous_preflight_without_domain_returns_disambiguation_guidance(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    result = run_eiel(
        [
            "bootstrap", "inspect",
            "--repo-root", str(repo),
            "--task-mode", "implementation",
            "--topic", "qualification authority register",
            "--operator-instruction-ref", "test",
            "--edit-authorized",
            "--write-scope", "tools",
            "--expected-validation", "pytest -q",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:11:00Z",
        ],
        state_root=state,
        expected=2,
    )
    assert result["status"] == "BLOCKED"
    bootstrap = result["bootstrap"]
    preflight = bootstrap["authority"]["preflight"]
    assert preflight["authority_resolution_state"] == "ambiguous"
    assert len(preflight.get("authority_domains") or []) > 1
    assert preflight.get("authority_assertion_ids")
    command = result["next_actions"][0]["command"]
    assert "--authority-domain" in command


def test_bootstrap_rejects_unknown_authority_domain(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    result = run_eiel(
        [
            "bootstrap", "inspect",
            "--repo-root", str(repo),
            "--task-mode", "implementation",
            "--topic", "qualification authority register",
            "--operator-instruction-ref", "test",
            "--edit-authorized",
            "--write-scope", "tools",
            "--expected-validation", "pytest -q",
            "--authority-domain", "not-a-real-authority-domain",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:12:00Z",
        ],
        state_root=state,
        expected=2,
    )
    assert result["status"] == "BLOCKED"
    preflight = result["bootstrap"]["authority"]["preflight"]
    assert preflight["authority_domain_filter"] == "not-a-real-authority-domain"
    assert preflight["authority_resolution_state"] == "none"


def test_bootstrap_no_store_diagnostic_creates_no_receipt_record(tmp_path: Path) -> None:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    result = run_eiel(
        [
            "bootstrap", "inspect",
            "--repo-root", str(repo),
            "--task-mode", "implementation",
            "--topic", "qualification authority register",
            "--operator-instruction-ref", "test",
            "--edit-authorized",
            "--write-scope", "tools",
            "--expected-validation", "pytest -q",
            "--authority-domain", "execution_route",
            "--no-store",
            "--state-root", str(state),
            "--generated-at", "2026-08-04T00:13:00Z",
        ],
        state_root=state,
    )
    assert result["record_sha256"] is None
    records = list((state / "bootstrap" / "records").glob("*.json"))
    assert not records
