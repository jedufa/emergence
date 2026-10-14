from __future__ import annotations

import csv
import json
from pathlib import Path

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.qualification_framework import (
    FULL_QUALIFICATION_REQUIRED,
    QUALIFIED,
    QUALIFIED_WITH_EXCLUSIONS,
    REVALIDATION_REQUIRED,
    ExclusionRegionV1,
    OperatingEnvelopeV1,
    load_catalog,
    plan_delta_campaign,
    resolve_qualification,
)
from tools.scripts import eiel_qualification


CATALOG = Path("_spec/qualification/qf1_synthetic_catalog_v1.json")
BASE_COMMIT = "e8ddd045791313f5cffbc4f61330274b248b88ac"
BASE_REQUEST = {
    "provider_id": "eiel-production-provider-v1",
    "runtime_id": "eiel-production-runtime-v1",
    "world_family": "patch",
    "execution_route_sha256": "route-synthetic-patch-v1",
    "source_commit": BASE_COMMIT,
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_run(
    root: Path,
    *,
    name: str,
    region_supported: int,
    route_receipt_present: int = 1,
    route_receipt_match: int = 1,
    no_integrity_failure: int = 1,
) -> Path:
    run = root / name
    run.mkdir(parents=True)
    fields = [
        "generation",
        "alive",
        "qf_route_receipt_present_flag",
        "qf_route_receipt_sha256_match_flag",
        "qf_route_binding_exact_flag",
        "qf_source_binding_exact_flag",
        "qf_coverage_complete_flag",
        "qf_repeat_parity_pass_flag",
        "qf_checkpoint_resume_pass_flag",
        "qf_finite_outputs_flag",
        "qf_no_integrity_failure_flag",
        "qf_no_posthoc_widening_flag",
        "qf_region_supported_flag",
    ]
    with (run / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "generation": 1,
                "alive": 1,
                "qf_route_receipt_present_flag": route_receipt_present,
                "qf_route_receipt_sha256_match_flag": route_receipt_match,
                "qf_route_binding_exact_flag": route_receipt_match,
                "qf_source_binding_exact_flag": 1,
                "qf_coverage_complete_flag": 1,
                "qf_repeat_parity_pass_flag": 1,
                "qf_checkpoint_resume_pass_flag": 1,
                "qf_finite_outputs_flag": 1,
                "qf_no_integrity_failure_flag": no_integrity_failure,
                "qf_no_posthoc_widening_flag": 1,
                "qf_region_supported_flag": region_supported,
            }
        )
    _write_json(run / "config.json", {"sim_cfg": {}, "env_cfg": {}})
    return run


def _manifest(root: Path, run: Path, *, campaign_id: str) -> Path:
    route_receipt = root / "route_receipt.json"
    _write_json(
        route_receipt,
        {
            "schema_version": "synthetic_route_receipt_v1",
            "execution_route_sha256": "route-synthetic-patch-v1",
            "source_commit": BASE_COMMIT,
        },
    )
    summary = root / "summary.json"
    _write_json(
        summary,
        {
            "runs": [
                {
                    "run_id": campaign_id,
                    "run_dir": str(run),
                    "status": "ok",
                    "returncode": 0,
                    "axes": {
                        "ecology_context": "qualification_delta",
                        "init_arm": "candidate",
                        "perception_condition": "normal",
                    },
                    "seed": 1,
                }
            ]
        },
    )
    manifest = root / "campaign_evidence_manifest.json"
    payload = {
            "branch": "integration/eiel-qualification-framework-v1",
            "campaign_id": campaign_id,
            "completed_run_count": 1,
            "config_files": [str(run / "config.json")],
            "config_glob": "config.json",
            "device": "synthetic",
            "failed_run_count": 0,
            "generations": 1,
            "git_commit": BASE_COMMIT,
            "grid_size": 24,
            "harness_manifest_paths": [],
            "harness_summary_paths": [str(summary)],
            "metrics_glob": "metrics.csv",
            "output_paths": [str(route_receipt)],
            "planned_run_count": 1,
            "population_size": 1,
            "profile": "QUAL-DELTA-1",
            "registry_path": str(manifest),
            "run_root": str(root),
            "scenario_matrix": {
                "dimensions": [
                    {"name": "ecology_context", "values": ["qualification_delta"]},
                    {"name": "init_arm", "values": ["candidate"]},
                    {"name": "perception_condition", "values": ["normal"]},
                ]
            },
            "seed_list": [1],
            "steps_per_generation": 1,
            "timestamp_utc": "1970-01-01T00:00:00Z",
            "worktree": str(root),
    }
    obligations_path = root / "qf_delta_obligations.json"
    if obligations_path.is_file():
        obligations = json.loads(obligations_path.read_text(encoding="utf-8"))
        payload.update(obligations)
    _write_json(manifest, payload)
    return manifest


def _run_lifecycle(root: Path) -> dict[str, str]:
    catalog = load_catalog(CATALOG)
    request = {
        **BASE_REQUEST,
        "grid_width": 24,
        "grid_height": 24,
        "ticks_per_generation": 120,
    }
    assert resolve_qualification(request, catalog)["decision"] == "DELTA_REQUIRED"
    plan = plan_delta_campaign(request, catalog.envelopes[0])
    plan_path = root / "plan.json"
    _write_json(plan_path, plan)
    _write_json(
        root / "qf_delta_obligations.json",
        {
            "qf_delta_required_obligations": plan["required_obligations"],
            "qf_delta_satisfied_obligations": plan["required_obligations"],
            "qf_delta_required_obligations_sha256": plan["required_obligations_sha256"],
        },
    )

    registry_path = root / "delta_registry.json"
    assert eiel_qualification.main(
        [
            "generate-registry",
            "--plan",
            str(plan_path),
            "--campaign-id",
            "qf1-synthetic-acceptance",
            "--output",
            str(registry_path),
        ]
    ) == 0
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["dry_run"]["ok"] is True

    assert eiel_qualification.main(["validate-plan", "--plan", str(plan_path)]) == 0

    manifest = _manifest(
        root,
        _write_run(root, name="run-a", region_supported=1),
        campaign_id="qf1-synthetic-acceptance",
    )
    decision_path = root / "qualification_decision.json"
    assert eiel_qualification.main(
        [
            "decide",
            "--manifest",
            str(manifest),
            "--profile",
            "QUAL-DELTA-1",
            "--plan-sha256",
            plan["plan_sha256"],
            "--source-commit",
            BASE_COMMIT,
            "--evidence-uri",
            "synthetic://qf1/acceptance",
            "--output",
            str(decision_path),
        ]
    ) == 0

    accepted_catalog_path = root / "accepted_catalog.json"
    assert eiel_qualification.main(
        [
            "generate-registry",
            "--catalog",
            str(CATALOG),
            "--plan",
            str(plan_path),
            "--decision",
            str(decision_path),
            "--envelope-id",
            "synthetic-w16-w32-t80-t160-candidate",
            "--output",
            str(accepted_catalog_path),
        ]
    ) == 0
    index_path = root / "index.json"
    assert eiel_qualification.main(
        ["index", "--evidence-root", str(root), "--output", str(index_path)]
    ) == 0
    first_index = index_path.read_text(encoding="utf-8")
    assert eiel_qualification.main(
        ["index", "--evidence-root", str(root), "--output", str(index_path)]
    ) == 0
    assert index_path.read_text(encoding="utf-8") == first_index
    return {
        "plan_sha256": plan["plan_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "decision_sha256": json.loads(decision_path.read_text(encoding="utf-8"))["evidence_sha256"],
        "accepted_catalog": str(accepted_catalog_path),
        "index_sha256": json.loads(index_path.read_text(encoding="utf-8"))["index_sha256"],
    }


def test_full_synthetic_lifecycle_publishes_scoped_qualified_envelope(tmp_path: Path) -> None:
    result = _run_lifecycle(tmp_path / "root-a")
    accepted = load_catalog(result["accepted_catalog"])
    for request in (
        {**BASE_REQUEST, "grid_width": 24, "grid_height": 24, "ticks_per_generation": 120},
        {**BASE_REQUEST, "grid_width": 18, "grid_height": 18, "ticks_per_generation": 100},
        {**BASE_REQUEST, "grid_width": 24, "grid_height": 32, "ticks_per_generation": 120},
    ):
        status = resolve_qualification(request, accepted)
        assert status["decision"] == QUALIFIED
        assert status["scope"] == "CONFIGURATION_OPERATING_ENVELOPE"


def test_full_synthetic_lifecycle_is_repeatable_across_clean_roots(tmp_path: Path) -> None:
    left = _run_lifecycle(tmp_path / "root-a")
    right = _run_lifecycle(tmp_path / "root-b")

    assert left["plan_sha256"] == right["plan_sha256"]
    assert left["registry_sha256"] == right["registry_sha256"]
    assert left["decision_sha256"] == right["decision_sha256"]
    assert canonical_sha256(load_catalog(left["accepted_catalog"]).payload()) == canonical_sha256(
        load_catalog(right["accepted_catalog"]).payload()
    )


def test_c_and_d_decisions_do_not_publish_envelope(tmp_path: Path) -> None:
    catalog = load_catalog(CATALOG)
    request = {
        **BASE_REQUEST,
        "grid_width": 24,
        "grid_height": 24,
        "ticks_per_generation": 120,
    }
    plan = plan_delta_campaign(request, catalog.envelopes[0])
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    for root in (tmp_path / "c", tmp_path / "d"):
        _write_json(
            root / "qf_delta_obligations.json",
            {
                "qf_delta_required_obligations": plan["required_obligations"],
                "qf_delta_satisfied_obligations": plan["required_obligations"],
                "qf_delta_required_obligations_sha256": plan["required_obligations_sha256"],
            },
        )

    for name, run in (
        ("c", _write_run(tmp_path / "c", name="run-c", region_supported=0)),
        ("d", _write_run(tmp_path / "d", name="run-d", region_supported=1, no_integrity_failure=0)),
    ):
        manifest = _manifest(tmp_path / name, run, campaign_id=f"qf1-synthetic-{name}")
        decision_path = tmp_path / name / "qualification_decision.json"
        assert eiel_qualification.main(
            [
                "decide",
                "--manifest",
                str(manifest),
                "--profile",
                "QUAL-DELTA-1",
                "--plan-sha256",
                plan["plan_sha256"],
                "--source-commit",
                BASE_COMMIT,
                "--evidence-uri",
                f"synthetic://qf1/{name}",
                "--output",
                str(decision_path),
            ]
        ) == 1
        assert eiel_qualification.main(
            [
                "generate-registry",
                "--catalog",
                str(CATALOG),
                "--plan",
                str(plan_path),
                "--decision",
                str(decision_path),
                "--envelope-id",
                "synthetic-w16-w32-t80-t160-candidate",
                "--output",
                str(tmp_path / name / "accepted_catalog.json"),
            ]
        ) == 2


def test_accepted_envelope_fail_closed_source_and_exclusion(tmp_path: Path) -> None:
    accepted = load_catalog(_run_lifecycle(tmp_path / "root")["accepted_catalog"])
    source_changed = resolve_qualification(
        {
            **BASE_REQUEST,
            "source_commit": "changed-source",
            "grid_width": 24,
            "grid_height": 24,
            "ticks_per_generation": 120,
        },
        accepted,
    )
    assert source_changed["decision"] == REVALIDATION_REQUIRED

    envelope = accepted.envelopes[0]
    excluded = type(accepted)(
        anchors=accepted.anchors,
        envelopes=(
            OperatingEnvelopeV1(
                envelope_id=envelope.envelope_id,
                scope=envelope.scope,
                axes=envelope.axes,
                categorical_invariants=envelope.categorical_invariants,
                anchor_ids=envelope.anchor_ids,
                required_corners=envelope.required_corners,
                accepted_points=envelope.accepted_points,
                evidence=envelope.evidence,
                source_commit=envelope.source_commit,
                execution_route_sha256=envelope.execution_route_sha256,
                exclusions=(
                    ExclusionRegionV1(
                        exclusion_id="synthetic-exclusion",
                        reason="synthetic excluded subregion",
                        axes=(),
                        categorical_values={},
                    ),
                ),
            ),
        ),
    )
    exclusion = resolve_qualification(
        {
            **BASE_REQUEST,
            "grid_width": 24,
            "grid_height": 24,
            "ticks_per_generation": 120,
        },
        excluded,
    )
    assert exclusion["decision"] == QUALIFIED_WITH_EXCLUSIONS
    assert resolve_qualification(
        {
            **BASE_REQUEST,
            "execution_route_sha256": "wrong-route",
            "grid_width": 24,
            "grid_height": 24,
            "ticks_per_generation": 120,
        },
        accepted,
    )["decision"] == FULL_QUALIFICATION_REQUIRED
