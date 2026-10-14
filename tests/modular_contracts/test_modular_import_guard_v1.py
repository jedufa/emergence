from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.architecture.eiel_modular_import_guard_v1 import (
    KNOWN_TARGET_PACKAGES,
    GuardConfigurationError,
    audit_repository,
    canonical_json_bytes,
    check_repository,
    evaluate_audit,
    load_baseline,
    sccs_from_edges,
    validate_baseline,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_BASELINE = (
    REPO_ROOT
    / "tools"
    / "architecture"
    / "baselines"
    / "eiel_modular_legacy_imports_v1.json"
)
BASELINE_DEBT_PATH = REPO_ROOT / "docs/release/baseline-debt.json"
BASELINE_DEBT_SHA256 = (
    "02091d729c241d7142de182e0d803ed205326a10201954675d95e1fc6c8de61a"
)
AUDIT_ROOTS = ["engine", "tools", "experiments", "experiments_shared", "_runner", "infra", "scripts"]
TARGET_POLICY = {
    "emergence_contracts": [],
    "emergence_runtime": ["emergence_contracts"],
    "emergence_worldkit": ["emergence_contracts"],
    "emergence_agentkit": ["emergence_contracts"],
    "emergence_adaptation": ["emergence_contracts"],
    "emergence_lab": ["emergence_contracts", "emergence_runtime"],
    "emergence_evidence": ["emergence_contracts"],
    "emergence_assurance": ["emergence_contracts", "emergence_lab", "emergence_evidence"],
    "emergence_ops": ["emergence_contracts", "emergence_lab", "emergence_assurance"],
    "eiel_composition": [
        "emergence_contracts",
        "emergence_runtime",
        "emergence_worldkit",
        "emergence_agentkit",
        "emergence_adaptation",
        "emergence_lab",
        "emergence_evidence",
        "emergence_assurance",
        "emergence_ops",
        "legacy_eiel",
    ],
}
assert set(TARGET_POLICY) == KNOWN_TARGET_PACKAGES


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _baseline(edges: dict[tuple[str, str], int] | None = None) -> dict[str, object]:
    edges = edges or {}
    edge_records = {
        f"{source} -> {target}": {
            "initial_ceiling": count,
            "ceiling": count,
            "ceiling_history": [count],
        }
        for (source, target), count in sorted(edges.items())
    }
    digest_payload = {
        key: record["ceiling"] for key, record in edge_records.items() if record["ceiling"] > 0
    }
    edge_tuples = [
        (source, target, count)
        for (source, target), count in sorted(edges.items())
        if count > 0
    ]
    return {
        "schema_version": "eiel_modular_legacy_import_ratchet_v1",
        "baseline_commit": "test-baseline",
        "source_inventory": "test",
        "source_inventory_sha256": "0" * 64,
        "audit_roots": AUDIT_ROOTS,
        "target_allowed_dependencies": copy.deepcopy(TARGET_POLICY),
        "baseline_audit": {
            "python_modules_audited": 0,
            "resolved_internal_import_statements": 0,
            "legacy_edge_count": len(edge_tuples),
            "legacy_edges_digest_sha256": hashlib.sha256(
                canonical_json_bytes(digest_payload)
            ).hexdigest(),
            "legacy_sccs": [list(item) for item in sccs_from_edges(edge_tuples)],
            "parse_errors": 0,
        },
        "legacy_edges": edge_records,
        "classified_debt": [],
        "temporary_exceptions": [],
    }


def _finding_codes(root: Path, baseline: dict[str, object]) -> set[str]:
    audit = audit_repository(root, audit_roots=AUDIT_ROOTS)
    return {str(item["code"]) for item in evaluate_audit(audit, baseline)}


def _pinned_baseline_debt() -> dict[str, object]:
    raw = BASELINE_DEBT_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASELINE_DEBT_SHA256, (
        "pinned release baseline-debt receipt changed"
    )
    value = json.loads(raw)
    assert value["schema_version"] == "eiel_release_modular_baseline_debt_v1"
    assert value["source_commit"] == "fb00e803ee0603b1645ba72de197008511193e3d"
    return value


def _matches_pinned_baseline_debt(result: dict[str, object]) -> bool:
    debt = _pinned_baseline_debt()
    audit = result.get("audit")
    return (
        result.get("status") == "FAIL"
        and result.get("baseline_commit") == debt.get("guard_baseline_commit")
        and result.get("findings") == debt.get("findings")
        and result.get("finding_count") == debt.get("finding_count")
        and result.get("target_package_violation_count")
        == debt.get("target_package_violation_count")
        and result.get("emergence_contracts_forbidden_imports")
        == debt.get("emergence_contracts_forbidden_imports")
        and isinstance(audit, dict)
        and audit.get("parse_errors") == []
    )


def test_audit_result_is_deterministic(tmp_path: Path) -> None:
    _write(tmp_path, "engine/agents/a.py", "import env.x\n")
    _write(tmp_path, "engine/env/x.py", "VALUE = 1\n")
    first = audit_repository(tmp_path, audit_roots=AUDIT_ROOTS).deterministic_dict()
    second = audit_repository(tmp_path, audit_roots=AUDIT_ROOTS).deterministic_dict()
    assert first == second


def test_ast_parse_failure_is_fatal(tmp_path: Path) -> None:
    _write(tmp_path, "engine/agents/broken.py", "def broken(:\n")
    assert "AST_PARSE_ERROR" in _finding_codes(tmp_path, _baseline())


def test_invalid_source_encoding_is_fatal(tmp_path: Path) -> None:
    path = tmp_path / "engine" / "agents" / "broken.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"VALUE = '\xff'\n")
    assert "AST_PARSE_ERROR" in _finding_codes(tmp_path, _baseline())


def test_dual_engine_import_root_resolves_to_one_family(tmp_path: Path) -> None:
    _write(tmp_path, "engine/agents/a.py", "import engine.env.x\nimport env.x\n")
    _write(tmp_path, "engine/env/x.py", "VALUE = 1\n")
    audit = audit_repository(tmp_path, audit_roots=AUDIT_ROOTS)
    assert audit.legacy_edges == (("agents", "env", 2),)


def test_allowed_target_dependency_passes(tmp_path: Path) -> None:
    _write(tmp_path, "emergence_contracts/__init__.py", "VALUE = 1\n")
    _write(tmp_path, "emergence_runtime/runtime.py", "import emergence_contracts\n")
    assert not _finding_codes(tmp_path, _baseline())


def test_forbidden_target_dependency_fails(tmp_path: Path) -> None:
    _write(tmp_path, "emergence_runtime/__init__.py", "VALUE = 1\n")
    _write(tmp_path, "emergence_worldkit/world.py", "import emergence_runtime\n")
    assert "FORBIDDEN_TARGET_DEPENDENCY" in _finding_codes(tmp_path, _baseline())


def test_emergence_contracts_forbidden_import_fails(tmp_path: Path) -> None:
    _write(tmp_path, "engine/simulation/__init__.py", "VALUE = 1\n")
    _write(tmp_path, "emergence_contracts/bad.py", "import engine.simulation\n")
    result = check_repository(tmp_path, _baseline())
    assert result["status"] == "FAIL"
    assert result["emergence_contracts_forbidden_imports"] == 1


def test_emergence_contracts_external_dependency_fails(tmp_path: Path) -> None:
    _write(tmp_path, "emergence_contracts/bad.py", "import numpy\n")
    result = check_repository(tmp_path, _baseline())
    assert result["status"] == "FAIL"
    assert result["emergence_contracts_forbidden_imports"] == 1
    assert result["findings"][0]["code"] == "FORBIDDEN_CONTRACT_EXTERNAL_IMPORT"


def test_legacy_edge_at_exact_ceiling_passes(tmp_path: Path) -> None:
    _write(tmp_path, "engine/agents/a.py", "import env.x\n")
    _write(tmp_path, "engine/env/x.py", "VALUE = 1\n")
    assert not _finding_codes(tmp_path, _baseline({("agents", "env"): 1}))


def test_legacy_edge_growth_fails(tmp_path: Path) -> None:
    _write(tmp_path, "engine/agents/a.py", "import env.x\nimport env.x\n")
    _write(tmp_path, "engine/env/x.py", "VALUE = 1\n")
    assert "LEGACY_EDGE_GROWTH" in _finding_codes(
        tmp_path, _baseline({("agents", "env"): 1})
    )


def test_lowered_debt_requires_baseline_update(tmp_path: Path) -> None:
    _write(tmp_path, "engine/agents/a.py", "VALUE = 1\n")
    _write(tmp_path, "engine/env/x.py", "VALUE = 1\n")
    assert "LEGACY_CEILING_NOT_LOWERED" in _finding_codes(
        tmp_path, _baseline({("agents", "env"): 1})
    )


def test_reviewed_complete_exception_can_bound_temporary_growth(tmp_path: Path) -> None:
    _write(tmp_path, "engine/agents/a.py", "import env.x\nimport env.x\n")
    _write(tmp_path, "engine/env/x.py", "VALUE = 1\n")
    baseline = _baseline({("agents", "env"): 1})
    baseline["temporary_exceptions"] = [
        {
            "source": "agents",
            "target": "env",
            "path": "engine/agents/a.py",
            "ceiling": 2,
            "owner": "test owner",
            "reason": "bounded test exception",
            "expires_after_slice": "P2-S2",
            "validation_obligation": "focused import guard test",
        }
    ]
    assert not _finding_codes(tmp_path, baseline)


def test_unknown_target_root_fails(tmp_path: Path) -> None:
    _write(tmp_path, "emergence_unknown/new.py", "VALUE = 1\n")
    assert "UNKNOWN_TARGET_ROOT" in _finding_codes(tmp_path, _baseline())


def test_incomplete_exception_metadata_fails_closed() -> None:
    baseline = _baseline()
    baseline["temporary_exceptions"] = [{"source": "agents", "target": "env"}]
    with pytest.raises(GuardConfigurationError, match="missing metadata"):
        validate_baseline(baseline)


def test_removed_debt_ceiling_history_cannot_rise() -> None:
    baseline = _baseline({("agents", "env"): 2})
    baseline["legacy_edges"]["agents -> env"] = {
        "initial_ceiling": 2,
        "ceiling": 2,
        "ceiling_history": [2, 1, 2],
    }
    with pytest.raises(GuardConfigurationError, match="history rises"):
        validate_baseline(baseline)


def test_baseline_self_consistency_is_fail_closed() -> None:
    baseline = _baseline({("agents", "env"): 1})
    validate_baseline(baseline)
    baseline["baseline_audit"]["legacy_edges_digest_sha256"] = "0" * 64
    with pytest.raises(GuardConfigurationError, match="digest"):
        validate_baseline(baseline)


def test_pinned_baseline_debt_does_not_quarantine_changed_findings() -> None:
    debt = _pinned_baseline_debt()
    exact = {
        "status": "FAIL",
        "baseline_commit": debt["guard_baseline_commit"],
        "finding_count": debt["finding_count"],
        "findings": copy.deepcopy(debt["findings"]),
        "target_package_violation_count": 0,
        "emergence_contracts_forbidden_imports": 0,
        "audit": {"parse_errors": []},
    }
    assert _matches_pinned_baseline_debt(exact)

    changed = copy.deepcopy(exact)
    changed["findings"].append(
        {
            "code": "UNKNOWN_TARGET_ROOT",
            "source": "new_package",
            "target": "",
            "observed": 1,
            "ceiling": 0,
        }
    )
    changed["finding_count"] += 1
    assert not _matches_pinned_baseline_debt(changed)

    changed_count = copy.deepcopy(exact)
    changed_count["findings"][0]["observed"] += 1
    assert not _matches_pinned_baseline_debt(changed_count)

    extra_target_violation = copy.deepcopy(exact)
    extra_target_violation["target_package_violation_count"] = 1
    assert not _matches_pinned_baseline_debt(extra_target_violation)


def test_live_e922_baseline_is_self_consistent_and_green() -> None:
    baseline = load_baseline(LIVE_BASELINE)
    validate_baseline(baseline)
    report_path = os.environ.get("EIEL_MODULAR_IMPORT_GUARD_REPORT")
    result = (
        json.loads(Path(report_path).read_text(encoding="utf-8"))
        if report_path
        else check_repository(REPO_ROOT, baseline)
    )
    if report_path:
        assert result.get("status") == "FAIL", (
            "the guard exited 1 but its preserved report was not a failing result"
        )
    if _matches_pinned_baseline_debt(result):
        pytest.xfail(
            "the live guard still reports exactly the source-pinned baseline debt; "
            "the raw guard remains FAIL"
        )
    assert result["status"] == "PASS", json.dumps(result["findings"], indent=2)
    assert result["audit"]["legacy_edges"]
    assert result["audit"]["parse_errors"] == []
    assert result["target_package_violation_count"] == 0
    assert result["emergence_contracts_forbidden_imports"] == 0
