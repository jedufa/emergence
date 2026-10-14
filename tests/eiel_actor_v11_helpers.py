from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

from tools import resource_monitor as erm
from tools.actors import authority
from tools.evidence.librarian import wp4

REPO_ROOT = Path(__file__).resolve().parents[1]
EIEL = REPO_ROOT / "tools/bin/eiel"
GIB = 1024**3
FIXED = "2026-08-03T12:00:00Z"


def run_eiel(
    args: Iterable[str],
    *,
    state_root: Path,
    expected: int | tuple[int, ...] = 0,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["EIEL_ACTOR_STATE_ROOT"] = str(state_root)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [str(EIEL), *list(args)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    accepted = (expected,) if isinstance(expected, int) else expected
    assert result.returncode in accepted, (
        f"command failed: {EIEL} {' '.join(args)}\n"
        f"returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    stream = result.stdout.strip() or result.stderr.strip()
    assert stream, f"command emitted no output: {' '.join(args)}"
    try:
        return json.loads(stream)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"command did not emit JSON: {' '.join(args)}\nstdout={result.stdout}\nstderr={result.stderr}"
        ) from exc


def write_safe_erm_report(path: Path, *, generated_at: str = FIXED, blocked: bool = False, gpu_available: bool = True) -> Path:
    blockers = []
    workloads = []
    gpu_processes = []
    if blocked:
        blockers = [{"class": "GPU_BLOCKER", "kind": "active_gpu_workload", "pid": 4242}]
        workloads = [{"pid": 4242, "device": "cuda", "campaign_id": "existing-campaign"}]
        gpu_processes = [{"pid": 4242, "used_memory_bytes": 3 * GIB}]
    report = {
        "schema_version": erm.SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "mode": "read_only_stateless_fresh_inspection",
        "non_goals": ["no_scheduler", "no_reservation", "no_launch", "no_process_kill"],
        "cpu": {"logical_threads": 8, "utilization_percent": 20.0},
        "memory": {"ram_available_bytes": 16 * GIB, "ram_total_bytes": 32 * GIB},
        "gpu": {
            "available": gpu_available,
            "gpus": ([{"index": 0, "name": "test-gpu", "vram_free_bytes": 5 * GIB, "vram_total_bytes": 6 * GIB}] if gpu_available else []),
            "processes": gpu_processes,
        },
        "storage": {"roots": [{"path": "/tmp/emergence-test/storage", "free_bytes": 100 * GIB}]},
        "running_eiel_workloads": workloads,
        "slot_heartbeats": [],
        "campaign_manifests": [],
        "ste": {"available": False},
        "blockers": blockers,
        "recommendation": {
            "value": "WAIT" if blocked else "SAFE",
            "confidence": "HIGH",
            "reason": "deterministic actor workflow fixture",
        },
        "confidence": "HIGH",
        "blocker_classification": ["GPU_BLOCKER"] if blocked else [],
        "recommended_poll_interval_seconds": 30,
        "next_check_after_seconds": 30,
    }
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return path


def write_preflight_receipt(
    path: Path,
    *,
    topic: str = "Codex actor manager bounded verification",
    authority_domain: str = "campaign_decision",
    fail_closed: bool = False,
    unknown_gaps: list[str] | None = None,
) -> Path:
    """Write a deterministic, catalog-bound preflight fixture.

    It is synthetic only in its search result rows.  Its catalog build and
    maintained-source digest are computed from the real repository catalog, so
    every actor process still verifies the current Librarian identity.
    """

    source_root = REPO_ROOT / "tools/evidence/librarian/catalog_source/00-catalog"
    digest = wp4.maintained_source_digest(source_root)
    build_id = "eiel-build-wp4-" + hashlib.sha256(digest.encode()).hexdigest()[:20]
    gaps = list(unknown_gaps or ([] if not fail_closed else ["fixture authority ambiguity"]))
    receipt = {
        "schema_version": authority.PREFLIGHT_RECEIPT_SCHEMA_VERSION,
        "topic": topic,
        "topic_terms": sorted(set(topic.lower().split())),
        "collection": "all",
        "authority_domain_filter": authority_domain,
        "catalog_build_id": build_id,
        "source_digest": digest,
        "authority_resolution_state": "resolved" if not fail_closed else "ambiguous",
        "fail_closed": bool(fail_closed),
        "fallback_repository_search_allowed": False,
        "authority_consulted": [{
            "authority_assertion_id": "eiel-auth-test-campaign-decision",
            "domain": authority_domain,
            "asserted_status": "test_bound_authority",
            "confidence": "explicit",
        }],
        "evidence_ids": ["EIEL-EV-000001"],
        "artifact_ids": ["eiel-art-test-0001"],
        "raw_asset_ids": [],
        "claims_consulted": ["eiel-claim-test-bounded"],
        "document_ids": ["EIEL-DOC-000005"],
        "document_relation_ids": [],
        "limitations_reviewed": ["eiel-limitation-test-bounded"],
        "unresolved_authority": bool(gaps),
        "unresolved_authority_gaps": gaps,
        "applicable_dossiers": ["dossier-test-bounded"],
        "next_step": "Proceed only within the bound operator instruction.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return path


def authorize_mission(
    mission_id: str,
    *,
    state_root: Path,
    ready: bool = True,
    generated_at: str = FIXED,
    fail_closed: bool = False,
    acknowledge_fail_closed: bool = False,
    authority_domain: str = "campaign_decision",
) -> dict[str, Any]:
    receipt = write_preflight_receipt(
        state_root / "fixtures" / f"{mission_id}-preflight.json",
        authority_domain=authority_domain,
        fail_closed=fail_closed,
    )
    instruction = state_root / "fixtures" / f"{mission_id}-operator-instruction.md"
    instruction.parent.mkdir(parents=True, exist_ok=True)
    instruction.write_text(
        f"# Test operator instruction\n\nBounded authority for {mission_id}.\n",
        encoding="utf-8",
    )
    args = [
        "missionPlanner", "preflight", mission_id,
        "--preflight-receipt", str(receipt),
        "--authorization-basis", "operator_instruction",
        "--operator-instruction-ref", str(instruction),
        "--repo-root", str(REPO_ROOT),
        "--state-root", str(state_root),
        "--generated-at", generated_at,
    ]
    if acknowledge_fail_closed:
        args.append("--acknowledge-fail-closed")
    if ready:
        args.append("--ready")
    return run_eiel(args, state_root=state_root, expected=0 if (not fail_closed or acknowledge_fail_closed) else 1)


def authorized_authority_context(
    tmp_path: Path,
    *,
    mission_type: str = "CONFIRMATION",
    device: str = "cpu",
) -> dict[str, Any]:
    receipt_path = write_preflight_receipt(tmp_path / "preflight.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    retained = authority.retain_preflight_receipt(receipt, state_root=tmp_path / "state")
    instruction = tmp_path / f"operator-instruction-{mission_type.lower()}-{device}.md"
    instruction.write_text(
        f"# Test operator authorization\n\nBounded to {mission_type}/{device}.\n",
        encoding="utf-8",
    )
    return authority.build_authority_context(
        authorization_basis="operator_instruction",
        repo_root=REPO_ROOT,
        state_root=tmp_path / "state",
        preflight_receipt=receipt,
        preflight_receipt_path=retained,
        operator_instruction_ref=str(instruction),
    )



def write_test_qualification_register(path: Path) -> Path:
    """Write a minimal self-contained technical authority register for actor tests.

    This fixture avoids depending on the full published documentation archive,
    which is intentionally absent from some core-source transfers. It grants
    authority only to the exact bounded test labels below.
    """

    tracks = [
        {
            "label": "b1w32p256c512gNt160cpu",
            "status": "qualified",
            "world": "patch",
            "grid": {"width": 32, "height": 32},
            "initial_population": 256,
            "resident_capacity": 512,
            "ticks_per_generation": 160,
            "execution_envelope": {"device": "cpu", "batch_cells": 1},
            "authority_scope": ["test_fixture_exact_route_only"],
            "limitations": ["synthetic actor-test authority only"],
            "durable_report_path": "tests/fixtures/qualification/b1w32-test-authority.json",
            "next_required_evidence": [],
        },
        {
            "label": "b48w16p64c128gNt160gpu",
            "status": "qualified",
            "world": "patch",
            "grid": {"width": 16, "height": 16},
            "initial_population": 64,
            "resident_capacity": 128,
            "ticks_per_generation": 160,
            "execution_envelope": {"device": "gpu", "batch_cells": 48},
            "authority_scope": ["test_fixture_exact_route_only"],
            "limitations": ["synthetic actor-test authority only"],
            "durable_report_path": "tests/fixtures/qualification/b48w16-test-authority.json",
            "next_required_evidence": [],
        },
    ]
    payload = {
        "schema_version": "eiel_qualification_authority_register_v1",
        "repository": {"commit": "test-fixture"},
        "policy": {"no_claims": "technical actor-test fixture only"},
        "tracks": tracks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path

def scientific_mission_spec() -> dict[str, Any]:
    return {
        "schema_version": "eiel_mission_specification_v2",
        "title": "Bounded perception confirmation workflow",
        "mission_type": "CONFIRMATION",
        "question": "Does the fixed COG-P1 fixture support its preregistered procedural criterion?",
        "decision_to_enable": "Accept or close the bounded fixture result without widening claims.",
        "scope": ["one fixed COG-P1 matched campaign"],
        "out_of_scope": ["mechanics changes", "coefficient tuning", "general cognition claims"],
        "controls": ["perception normal", "perception ablated"],
        "success_criteria": ["the versioned COG-P1 rule returns A"],
        "failure_criteria": ["the versioned COG-P1 rule returns B, C, or D"],
        "stopping_rules": ["stop after the fixed assessment; do not tune or rerun to rescue the result"],
        "decision_rule": "Apply the implemented COG-P1 profile exactly as versioned.",
        "claim_ceiling": "Procedural COG-P1 fixture decision only; no general cognition claim.",
        "required_evidence": ["paired outputs", "reward-leakage audit", "technical integrity record"],
        "technical_qualification_requirements": ["current b1w32p256c512gNt160cpu authority"],
        "qualification_request": {"track_label": "b1w32p256c512gNt160cpu"},
        "resource_demand": {
            "schema_version": "eiel_resource_demand_v1",
            "device": "cpu",
            "exclusive_gpu": False,
            "minimum_ram_available_gib": 4.0,
            "minimum_vram_available_gib": 0.0,
            "minimum_storage_available_gib": 1.0,
            "expected_duration_minutes": 5,
            "workload_class": "bounded_fixture_assessment",
        },
        "tasks": ["check route authority", "assess resources", "assess fixed outputs", "close mission"],
        "required_closeout_refs": [
            "evidence_refs",
            "qualification_refs",
            "resource_refs",
            "report_refs",
            "campaign_output_refs",
            "source_identity",
            "librarian_document_ids",
        ],
    }
