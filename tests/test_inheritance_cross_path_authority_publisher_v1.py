from __future__ import annotations

from pathlib import Path

import pytest

from engine.campaigns.inheritance_cross_path_authority_publisher_v1 import (
    CampaignSpecArtifactBindingRequestV1,
    InheritanceCrossPathAuthorityPublicationError,
    InheritanceCrossPathAuthorityPublicationRequestV1,
    bind_campaign_spec_artifacts_v1,
    publish_inheritance_cross_path_authority_v1,
)


def test_retired_authority_publication_always_fails_before_io(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "authority"
    request = InheritanceCrossPathAuthorityPublicationRequestV1(
        exact_full_candidate_binding_path=tmp_path / "exact.json",
        compact_candidate_binding_path=tmp_path / "compact.json",
        mixed_gate_plan_path=tmp_path / "mixed-plan.json",
        mixed_capture_manifest_path=tmp_path / "mixed-capture.json",
        mixed_gate_result_path=tmp_path / "mixed-result.json",
        cross_storage_plan_path=tmp_path / "storage-plan.json",
        cross_storage_capture_manifest_path=tmp_path / "storage-capture.json",
        cross_storage_result_path=tmp_path / "storage-result.json",
        reporting_enabled_evidence_path=tmp_path / "reporting-on.json",
        reporting_disabled_evidence_path=tmp_path / "reporting-off.json",
        execution_envelope_path=tmp_path / "execution.json",
        output_dir=output_dir,
    )

    with pytest.raises(
        InheritanceCrossPathAuthorityPublicationError,
        match="P12_RETIRED_INHERITANCE_AUTHORITY",
    ):
        publish_inheritance_cross_path_authority_v1(request)

    assert not output_dir.exists()


def test_retired_authority_cannot_bind_campaign_spec(tmp_path: Path) -> None:
    output_spec = tmp_path / "bound.yaml"
    request = CampaignSpecArtifactBindingRequestV1(
        source_spec_path=tmp_path / "source.yaml",
        production_binding_path=tmp_path / "binding.json",
        output_spec_path=output_spec,
        expected_source_sha256="0" * 64,
    )

    with pytest.raises(
        InheritanceCrossPathAuthorityPublicationError,
        match="P12_RETIRED_INHERITANCE_AUTHORITY",
    ):
        bind_campaign_spec_artifacts_v1(request)

    assert not output_spec.exists()
