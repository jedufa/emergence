from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eiel_composition.adapters import (
    LegacyEvidenceEvaluatorAdapter,
    LegacyRunArtifactBindingV1,
    LegacyRunArtifactManifestAdapter,
)
from emergence_contracts import (
    AssemblyId,
    CellId,
    CanonicalPayloadV1,
    ContentDigest,
    ContractError,
    EvidenceEvaluatorV1,
    EvidenceProfileV1,
    RunCompletionStatusV1,
    RunId,
    SchemaVersion,
)
from tests.evidence.cogp1_fixtures import cogp1_manifest
from tools.evidence.contracts import check_campaign_schema
from tools.evidence.decisions import analyze_profile
from tools.evidence.reports import generate_campaign_decision_report


REPORT_FILES = (
    "anomaly_summary.json",
    "campaign_evidence_manifest.json",
    "decision_report.md",
    "decision_summary.json",
    "paired_comparisons.csv",
    "run_status.csv",
    "schema_status.json",
)


def _binding() -> LegacyRunArtifactBindingV1:
    return LegacyRunArtifactBindingV1(
        run_id=RunId("phaseIV_cog_p1_perception_selection_probe"),
        cell_id=CellId("campaign-aggregate"),
        assembly_id=AssemblyId("legacy-eiel-assembly"),
        assembly_digest=ContentDigest("sha256", "a" * 64),
        runtime_route_id="legacy.campaign.evidence",
        source_versions=("git:abc",),
        output_schema_versions=(SchemaVersion("legacy_campaign_evidence_manifest_v1"),),
    )


def _profile() -> EvidenceProfileV1:
    return EvidenceProfileV1(
        profile_id="COG-P1",
        profile_version=SchemaVersion("cog_p1_v1"),
        rules=CanonicalPayloadV1.from_value(
            SchemaVersion("legacy_evidence_profile_rules_v1"),
            {"legacy_profile": "COG-P1"},
        ),
    )


def _tree_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {
        str(path.relative_to(root)): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    }


def test_legacy_manifest_adapter_preserves_current_manifest_artifacts(
    tmp_path: Path,
) -> None:
    manifest_path = cogp1_manifest(tmp_path / "input", signal="supported")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_path = Path(raw["harness_summary_paths"][0]).resolve()
    adapter = LegacyRunArtifactManifestAdapter()

    manifest = adapter.adapt(manifest_path, _binding())

    assert manifest.run_id == _binding().run_id
    assert manifest.completion_status is RunCompletionStatusV1.COMPLETE
    assert manifest.checkpoint_refs == ()
    assert manifest.event_refs == ()
    assert manifest.metric_refs[0].locator == str(manifest_path.resolve())
    assert str(summary_path) in {item.locator for item in manifest.metric_refs}
    assert len(manifest.metric_refs) == 122
    assert len(manifest.config_digests) == 120
    for reference in manifest.metric_refs:
        payload = Path(reference.locator).read_bytes()
        assert reference.size_bytes == len(payload)
        assert reference.content_digest.hex == hashlib.sha256(payload).hexdigest()
    assert manifest == type(manifest).from_dict(manifest.to_dict())


def test_evaluator_direct_and_adapter_results_are_exact(tmp_path: Path) -> None:
    manifest_path = cogp1_manifest(tmp_path / "input", signal="supported")
    manifest_adapter = LegacyRunArtifactManifestAdapter()
    evaluator = LegacyEvidenceEvaluatorAdapter()
    assert isinstance(evaluator, EvidenceEvaluatorV1)
    evaluator.ingest(manifest_adapter.reference_evidence_manifest(manifest_path))

    direct_status = check_campaign_schema(manifest_path)
    direct_decision = analyze_profile(direct_status, "COG-P1")
    evaluation = evaluator.decide(_profile())

    assert evaluation.validation.valid is bool(direct_status["ok"])
    assert evaluation.comparison.to_value() == direct_decision
    assert evaluation.decision.to_value() == {
        key: direct_decision[key]
        for key in (
            "profile",
            "decision",
            "rationale",
            "claims_allowed",
            "claims_not_allowed",
            "recommended_next_action",
        )
    }


def test_deterministic_report_matches_legacy_generator_byte_for_byte(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    manifest_path = cogp1_manifest(input_root, signal="supported")
    before = _tree_snapshot(input_root)
    direct_output = tmp_path / "direct-report"
    adapter_output = tmp_path / "adapter-report"
    generate_campaign_decision_report(
        manifest_path, "COG-P1", direct_output, deterministic=True
    )

    evaluator = LegacyEvidenceEvaluatorAdapter()
    evaluator.ingest(
        LegacyRunArtifactManifestAdapter().reference_evidence_manifest(manifest_path)
    )
    evaluator.decide(_profile())
    request = evaluator.report_output_request(adapter_output)
    bundle = evaluator.materialize_report(request)

    assert bundle.locator == str(adapter_output.resolve())
    assert bundle.size_bytes == sum(
        (adapter_output / name).stat().st_size for name in REPORT_FILES
    )
    for filename in REPORT_FILES:
        assert (adapter_output / filename).read_bytes() == (
            direct_output / filename
        ).read_bytes()
    assert _tree_snapshot(input_root) == before


def test_ingest_fails_closed_if_referenced_manifest_changes(tmp_path: Path) -> None:
    manifest_path = cogp1_manifest(tmp_path / "input", signal="supported")
    reference = LegacyRunArtifactManifestAdapter().reference_evidence_manifest(
        manifest_path
    )
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )

    with pytest.raises(ContractError, match="immutable file content"):
        LegacyEvidenceEvaluatorAdapter().ingest(reference)


def test_evidence_adapter_imports_no_simulator_implementation() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    code = f"""
import json, sys
sys.path.insert(0, {str(repository_root)!r})
import eiel_composition.adapters.legacy_evidence
forbidden = {{'agents', 'env', 'evolution', 'native_runtime', 'perf', 'simulation'}}
loaded = sorted(forbidden.intersection(name.split('.', 1)[0] for name in sys.modules))
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []
