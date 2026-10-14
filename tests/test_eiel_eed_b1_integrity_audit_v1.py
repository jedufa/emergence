from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from tools.scripts.eiel_eed_b1_integrity_audit_v1 import (
    EedB1IntegrityError,
    attempt_identity,
    classify_impact_archive_file,
    classify_phenotype_attempt_file,
    is_reducer_input_path,
    public_checkpoint_admitted,
    recursive_inventory,
    treatment_semantic_isolation,
    verify_input_manifest,
)


TREATMENTS = ("impact_on", "impact_off", "ecology_frozen", "phenotype_fixed")


def _spec(treatment: str) -> dict[str, object]:
    values = {
        "impact_on": (True, False, False),
        "impact_off": (False, False, False),
        "ecology_frozen": (True, True, False),
        "phenotype_fixed": (True, False, True),
    }
    impact, ecology, phenotype = values[treatment]
    return {
        "campaign_id": f"eed-b1-science-{treatment}-g100",
        "description": f"frozen {treatment}",
        "parameters": {
            "eed": {
                "resource_impact_apply_enable": impact,
                "ecology_frozen": ecology,
                "phenotype_fixed": phenotype,
            }
        },
        "artifacts": {
            name: {"path": f"/route/{treatment}/{name}.json"}
            for name in (
                "publication_receipt",
                "qualification_receipt",
                "reporting_receipt",
            )
        },
        "output": {"runtime_root": f"/runs/{treatment}"},
        "shared": {"unchanged": 1},
    }


def _write_specs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for treatment in TREATMENTS:
        (root / f"eed-b1-science-{treatment}-g100.yaml").write_text(
            yaml.safe_dump(_spec(treatment), sort_keys=False),
            encoding="utf-8",
        )


def test_treatment_semantic_isolation_accepts_only_frozen_differences(
    tmp_path: Path,
) -> None:
    _write_specs(tmp_path)

    result = treatment_semantic_isolation(tmp_path)

    assert result["status"] == "pass"
    assert result["forbidden_or_missing_difference_count"] == 0
    assert (
        result["pairs"]["impact_off__ecology_frozen"]["interpretation"]
        == "descriptive_control_contrast"
    )


def test_treatment_semantic_isolation_fails_on_extra_difference(
    tmp_path: Path,
) -> None:
    _write_specs(tmp_path)
    path = tmp_path / "eed-b1-science-impact_off-g100.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec["shared"]["unchanged"] = 2
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    with pytest.raises(EedB1IntegrityError, match="semantic isolation failed"):
        treatment_semantic_isolation(tmp_path)


@pytest.mark.parametrize(
    ("relative", "content", "classifier", "classification"),
    [
        (
            "uninterrupted_baseline/cells/c/segment/metrics.csv",
            "generation,alive\n",
            classify_impact_archive_file,
            "setup_header_only_metrics_csv",
        ),
        (
            "uninterrupted_baseline/cells/c/segment/metrics/lineage_summary.csv",
            "lineage,count\n",
            classify_impact_archive_file,
            "setup_header_only_lineage_csv",
        ),
        (
            "uninterrupted_baseline/cells/c/segment/metrics/lineage_events.jsonl",
            "",
            classify_impact_archive_file,
            "setup_empty_jsonl_surface",
        ),
        (
            "cells/c/segment/metrics.csv",
            "generation,alive\n",
            classify_phenotype_attempt_file,
            "setup_header_only_metrics_csv",
        ),
    ],
)
def test_zero_decision_output_setup_surfaces_are_accepted(
    tmp_path: Path,
    relative: str,
    content: str,
    classifier: object,
    classification: str,
) -> None:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    result = classifier(Path(relative), path)

    assert result["classification"] == classification


@pytest.mark.parametrize(
    ("relative", "content"),
    [
        (
            "uninterrupted_baseline/cells/c/segment/metrics.csv",
            "generation,alive\n0,64\n",
        ),
        (
            "uninterrupted_baseline/cells/c/segment/metrics/lineage_events.jsonl",
            '{"generation": 0}\n',
        ),
        ("checkpoint_bundles_v2/c/bundle_gen000050/metadata.json", "{}\n"),
        ("native_runtime_checkpoint_projection.json.gz", "not-empty"),
        ("result.json", "{}\n"),
        ("unknown.bin", "not-empty"),
    ],
)
def test_decision_output_or_unknown_nonempty_archive_file_is_rejected(
    tmp_path: Path,
    relative: str,
    content: str,
) -> None:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    with pytest.raises(EedB1IntegrityError):
        classify_impact_archive_file(Path(relative), path)


def test_staged_checkpoint_is_classified_but_not_publicly_admitted(
    tmp_path: Path,
) -> None:
    staged = (
        tmp_path
        / "checkpoint_bundles_v2"
        / "cell"
        / ".bundle_gen000050.tmp-test"
        / "metadata.json"
    )
    staged.parent.mkdir(parents=True)
    staged.write_text("{}\n", encoding="utf-8")
    public = staged.parent.parent / "bundle_gen000050"

    classification = classify_phenotype_attempt_file(
        staged.relative_to(tmp_path),
        staged,
    )

    assert classification["classification"] == "unpublished_checkpoint_staging_payload"
    assert not public_checkpoint_admitted({"checkpoint": None}, public)
    assert not is_reducer_input_path(tmp_path, staged)


def test_attempt_identities_distinguish_superseded_and_fresh_attempts() -> None:
    assert attempt_identity("batch-0001", 1) != attempt_identity("batch-0001", 2)


def test_reducer_input_manifest_rejects_excluded_attempt_path(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "batch_attempt_recovery_archive"
        / "batch-0001-attempt1"
        / "manifest.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "included_file_count": 1,
        "inputs": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "included_in_reduction": True,
            }
        ],
    }

    with pytest.raises(EedB1IntegrityError, match="excluded reducer input"):
        verify_input_manifest(manifest, evidence_root=tmp_path)


def test_recursive_inventory_is_deterministic_and_read_only(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text("", encoding="utf-8")
    before = path.read_bytes()

    first = recursive_inventory(
        tmp_path,
        classifier=classify_impact_archive_file,
    )
    second = recursive_inventory(
        tmp_path,
        classifier=classify_impact_archive_file,
    )

    assert first["canonical_inventory_sha256"] == second["canonical_inventory_sha256"]
    assert path.read_bytes() == before


def test_public_checkpoint_requires_exact_published_bundle_path(
    tmp_path: Path,
) -> None:
    public = tmp_path / "bundle_gen000050"
    public.mkdir()
    manifest = {
        "checkpoint": {
            "cell_bundles": [
                {"cell_id": "cell", "path": str(public.resolve())},
            ]
        }
    }

    assert public_checkpoint_admitted(manifest, public)
    assert not public_checkpoint_admitted(manifest, tmp_path / "other")


def test_frozen_manifest_verification_fails_on_changed_input(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "included_file_count": 1,
        "inputs": [
            {
                "path": str(input_path),
                "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "included_in_reduction": True,
            }
        ],
    }
    input_path.write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(EedB1IntegrityError, match="input changed"):
        verify_input_manifest(manifest, evidence_root=tmp_path)
