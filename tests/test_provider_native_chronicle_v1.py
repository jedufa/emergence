from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from engine.campaigns.chronicle_closeout import (
    CampaignChronicleCloseoutError,
    materialize_completed_campaign_chronicles,
)
from engine.campaigns.provider_native_chronicle_v1 import (
    CONTINUATION_REBIND_SCHEMA_VERSION,
    ProviderNativeChronicleConsumerV1,
    ProviderNativeChronicleError,
    ProviderNativeChronicleReceiptV1,
)
from engine.campaigns.production_engine_identity import PRODUCTION_PROVIDER_ID
from engine.perf.cuda_architecture_b_resident_reproduction_exposure_sidecar_v1 import (
    CONTINUITY_SCHEMA_VERSION,
)
from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
    SOURCE_IMPORT_CURSOR_SCHEMA_VERSION_V1,
    DetachedSourcePayloadV1,
    SourceImportCursorV1,
)
from engine.perf.cuda_genome_upgrade_v2_state_bank import (
    CudaGenomeUpgradeV2StateBank,
)
from engine.perf.inheritance_sidecar_identity_v1 import (
    INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1,
    inheritance_sidecar_payload_sha256_v1,
)
from evolution.genome_upgrade_v2_telemetry import (
    summarize_genome_upgrade_v2_resident_tensors,
)
from simulation.chronicle.schema import canonical_json
from simulation.chronicle.storage import ARTIFACT_NAMES


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _artifact(
    cell_id: str,
    *,
    segment_index: int = 0,
    storage_mode: str = "exact_full",
    path: int = 0,
    previous_chain_sha256: str = "0" * 64,
) -> dict:
    attempt = {
        "attempt_index": 0,
        "path": path,
        "generation": segment_index,
        "tick": 4,
        "eligible_parent_counts": [3, 2],
        "selected_parent_ids": [11, 13],
        "success": True,
        "failure_code": 0,
        "commit_ordinal": 7,
        "authoritative_commit": {
            "child_agent_id": 17,
            "commit_authority_sha256": "c" * 64,
        },
    }
    snapshots = []
    if storage_mode == "exact_full":
        attempt["eligible_snapshot_refs"] = [0, 1]
        snapshots = [{"snapshot_index": 0}, {"snapshot_index": 1}]
    cell = {
        "cell_index": 0,
        "enabled": True,
        "status": 0,
        "attempt_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "snapshot_count": len(snapshots),
        "attempts": [attempt],
        "eligible_parent_snapshots": snapshots,
        "selected_parent_counts": [{"agent_id": 11, "count": 1}],
        "parent_contribution_counts": [{"agent_id": 11, "count": 1}],
        "primary_eligibility_opportunity_counts": [
            {"agent_id": 11, "count": 1}
        ],
        "mate_eligibility_opportunity_counts": [{"agent_id": 13, "count": 1}],
        "cumulative_counts_after_segment": {
            "attempts": segment_index + 1,
            "successes": segment_index + 1,
            "failures": 0,
            "eligible_snapshots": (
                2 * (segment_index + 1) if storage_mode == "exact_full" else 0
            ),
        },
    }
    schema = "inheritance_cross_path_reproduction_exposure_segment_v1"
    cell["segment_sha256"] = _digest(
        {"schema_version": schema, "segment_index": segment_index, "cell": cell}
    )
    cell["previous_chain_sha256"] = previous_chain_sha256
    cell["chain_sha256_after_segment"] = _digest(
        {
            "schema_version": CONTINUITY_SCHEMA_VERSION,
            "segment_index": segment_index,
            "previous_chain_sha256": previous_chain_sha256,
            "segment_sha256": cell["segment_sha256"],
        }
    )
    return {
        "schema_version": schema,
        "segment_index": segment_index,
        "capacity_semantics": {"storage_mode": storage_mode},
        "cell_id": cell_id,
        "cell": cell,
        "segment_output_sha256": str(segment_index + 1) * 64,
    }


def _consumer(root: Path, cell_ids: tuple[str, ...] = ("cell-a",)):
    return ProviderNativeChronicleConsumerV1(
        batch_root=root,
        campaign_id="campaign-a",
        batch_id="batch-a",
        cell_ids=cell_ids,
        profile="scientific_full",
    )


def _observe(
    consumer: ProviderNativeChronicleConsumerV1,
    artifacts: tuple[dict, ...],
    *,
    segment_index: int = 0,
    checkpoint_projections: tuple[dict, ...] | None = None,
) -> None:
    consumer.observe_segment(
        segment_ordinal=1,
        start_generation=segment_index,
        end_generation=segment_index + 1,
        cell_ids=tuple(row["cell_id"] for row in artifacts),
        segment_artifacts=artifacts,
        checkpoint_projections=(
            checkpoint_projections
            if checkpoint_projections is not None
            else tuple({} for _ in artifacts)
        ),
    )


def _projection_cursor(
    payloads: tuple[dict, ...], cell_ids: tuple[str, ...]
) -> SourceImportCursorV1:
    assert len(payloads) == len(cell_ids)

    def loader(_descriptor: object, index: int, cell_id: str) -> DetachedSourcePayloadV1:
        payload = payloads[index]
        return DetachedSourcePayloadV1(
            schema_version=SOURCE_IMPORT_CURSOR_SCHEMA_VERSION_V1,
            cell_id=cell_id,
            cell_index=index,
            payload=payload,
            source_projection_sha256=_digest(payload),
            live_payload_bytes=1,
        )

    return SourceImportCursorV1.begin(
        tuple(range(len(cell_ids))),
        cell_ids,
        loader=loader,
        importer=lambda _payload, _storage, _index, _cell_id: None,
    )


def _genome_v2_checkpoint(
    cell_id: str,
    *,
    generation: int = 1,
    tick: int = 0,
) -> dict:
    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=1, capacity=2, device="cpu"
    )
    bank.products_q10[0, 0, 0] = 1024
    payload = bank.checkpoint_projection()
    payload["compact_generation_summary"] = (
        summarize_genome_upgrade_v2_resident_tensors(
            bank.named_tensors(),
            alive_mask=torch.tensor([[True, False]], dtype=torch.bool),
        )
    )
    return {
        "genome_upgrade_v2_checkpoint_projection": {
            **payload,
            "identity": {
                "schema_version": INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1,
                "cell_id": cell_id,
                "seed": 41,
                "generation": generation,
                "tick": tick,
                "base_projection_sha256": "b" * 64,
                "payload_sha256": (
                    inheritance_sidecar_payload_sha256_v1(payload)
                ),
            },
        }
    }


def _genome_v2_checkpoint_digest(checkpoint: dict) -> str:
    return checkpoint["genome_upgrade_v2_checkpoint_projection"][
        "identity"
    ]["payload_sha256"]


def test_exact_chronicle_represents_both_reproduction_paths_and_source_binding(
    tmp_path: Path,
) -> None:
    consumer = _consumer(tmp_path)
    first = _artifact("cell-a", path=0)
    second = _artifact(
        "cell-a",
        segment_index=1,
        path=1,
        previous_chain_sha256=first["cell"]["chain_sha256_after_segment"],
    )
    _observe(consumer, (first,))
    _observe(consumer, (second,), segment_index=1)
    receipt = consumer.finalize().payload()

    root = Path(receipt["cell_receipts"]["cell-a"]["chronicle_root"])
    events = [
        json.loads(line)
        for line in (root / "formal_events.jsonl").read_text().splitlines()
    ]
    assert [row["measurements"]["reproduction_path_id"] for row in events] == [0, 1]
    assert [row["measurements"]["provider_native_event_cursor"] for row in events] == [
        0,
        1,
    ]
    assert all(
        row["measurements"]["eligible_parent_snapshot_claims_present"] is True
        for row in events
    )
    assert receipt["cell_receipts"]["cell-a"]["segment_bindings"][1] == {
        "segment_index": 1,
        "start_generation": 1,
        "end_generation": 2,
        "source_segment_sha256": "2" * 64,
        "cell_segment_sha256": second["cell"]["segment_sha256"],
        "previous_chain_sha256": first["cell"]["chain_sha256_after_segment"],
        "chain_sha256_after_segment": second["cell"][
            "chain_sha256_after_segment"
        ],
    }


def test_compact_chronicle_never_fabricates_snapshot_claims(tmp_path: Path) -> None:
    consumer = _consumer(tmp_path)
    _observe(consumer, (_artifact("cell-a", storage_mode="compact_complete"),))
    receipt = consumer.finalize().payload()
    root = Path(receipt["cell_receipts"]["cell-a"]["chronicle_root"])
    event = json.loads((root / "formal_events.jsonl").read_text().strip())

    measurements = event["measurements"]
    assert measurements["eligible_parent_snapshot_claims_present"] is False
    assert "eligible_snapshot_refs" not in measurements


def test_segment_consumer_accepts_one_cell_projection_cursor(tmp_path: Path) -> None:
    cell_ids = ("cell-a", "cell-b")
    artifacts = (
        _artifact("cell-a", path=0),
        _artifact("cell-b", path=1),
    )
    cursor = _projection_cursor(({}, {}), cell_ids)
    consumer = _consumer(tmp_path, cell_ids)

    consumer.observe_segment(
        segment_ordinal=1,
        start_generation=0,
        end_generation=1,
        cell_ids=cell_ids,
        segment_artifacts=artifacts,
        checkpoint_projections=None,
        checkpoint_projection_cursor=cursor,
    )

    receipt = cursor.close().payload()
    assert receipt["state"] == "SEALED"
    assert receipt["covered_cell_ids"] == list(cell_ids)
    assert receipt["max_live_tensor_bearing_payloads"] == 1
    assert receipt["importer_concurrency"] == 1


def test_generation_zero_consumer_accepts_one_cell_projection_cursor(
    tmp_path: Path,
) -> None:
    cell_id = "cell-a"
    projection = _genome_v2_checkpoint(cell_id, generation=0)
    source_digest = _genome_v2_checkpoint_digest(projection)

    def loader(
        _descriptor: object, index: int, loaded_cell_id: str
    ) -> DetachedSourcePayloadV1:
        return DetachedSourcePayloadV1(
            schema_version=SOURCE_IMPORT_CURSOR_SCHEMA_VERSION_V1,
            cell_id=loaded_cell_id,
            cell_index=index,
            payload=projection,
            source_projection_sha256=source_digest,
            live_payload_bytes=1,
        )

    cursor = SourceImportCursorV1.begin(
        (0,),
        (cell_id,),
        loader=loader,
        importer=lambda _payload, _storage, _index, _cell_id: None,
    )
    consumer = _consumer(tmp_path)

    consumer.observe_generation_zero(
        cell_ids=(cell_id,),
        checkpoint_projections=None,
        checkpoint_projection_cursor=cursor,
        source_projection_sha256_by_cell={cell_id: source_digest},
        source_projection_digest=_genome_v2_checkpoint_digest,
    )

    receipt = cursor.close().payload()
    assert receipt["state"] == "SEALED"
    assert receipt["covered_cell_ids"] == [cell_id]
    assert receipt["max_live_tensor_bearing_payloads"] == 1


def test_fresh_consumer_replay_is_idempotent_and_continuation_has_no_gap(
    tmp_path: Path,
) -> None:
    first = _artifact("cell-a")
    _observe(_consumer(tmp_path), (first,))
    resumed = _consumer(tmp_path)
    _observe(resumed, (first,))
    second = _artifact(
        "cell-a",
        segment_index=1,
        path=1,
        previous_chain_sha256=first["cell"]["chain_sha256_after_segment"],
    )
    _observe(
        resumed,
        (second,),
        segment_index=1,
    )
    receipt = resumed.finalize().payload()["cell_receipts"]["cell-a"]

    assert receipt["segment_count"] == 2
    assert receipt["event_cursor"] == 2
    with pytest.raises(ProviderNativeChronicleError, match="replayed"):
        changed = _artifact("cell-a")
        changed["segment_output_sha256"] = "9" * 64
        _observe(_consumer(tmp_path), (changed,))


def test_multicell_roots_are_unique_and_reporting_is_observational(
    tmp_path: Path,
) -> None:
    roots = []
    payloads = []
    for reporting in ("enabled", "disabled"):
        batch_root = tmp_path / reporting
        consumer = _consumer(batch_root, ("cell-a", "cell-b"))
        _observe(
            consumer,
            (
                _artifact("cell-a", path=0),
                _artifact("cell-b", path=1),
            ),
        )
        receipt = consumer.finalize().payload()
        cell_roots = [
            Path(row["chronicle_root"])
            for row in receipt["cell_receipts"].values()
        ]
        assert len(set(cell_roots)) == 2
        roots.append(cell_roots)
        payloads.append(
            [
                (root / "formal_events.jsonl").read_bytes()
                for root in cell_roots
            ]
        )
    assert payloads[0] == payloads[1]
    assert roots[0] != roots[1]


def test_genome_v2_checkpoint_summary_is_passively_projected(
    tmp_path: Path,
) -> None:
    checkpoint = _genome_v2_checkpoint("cell-a")
    checkpoint_before = json.dumps(
        checkpoint,
        sort_keys=True,
        default=lambda value: value.tolist(),
    )
    consumer = _consumer(tmp_path)
    _observe(
        consumer,
        (_artifact("cell-a"),),
        checkpoint_projections=(checkpoint,),
    )
    receipt = consumer.finalize().payload()
    root = Path(receipt["cell_receipts"]["cell-a"]["chronicle_root"])
    report = json.loads((root / ARTIFACT_NAMES["summary"]).read_text())
    projected = report["genome_v2_generation_summaries"]

    assert len(projected) == 1
    assert projected[0]["cell_id"] == "cell-a"
    assert projected[0]["summary"]["raw_tick_arrays_retained"] is False
    hereditary = projected[0]["summary"]["cells"][0][
        "hereditary_capacity"
    ]
    assert hereditary["schema_version"] == (
        "genome_v2_hereditary_capacity_summary_v1"
    )
    assert hereditary["locus_summary_count"] == 60
    assert hereditary["raw_genotype_rows_retained"] is False
    assert (
        projected[0]["summary"]["cells"][0]["products_q10"]["energy_flux"][
            "saturation_frequency"
        ]
        == 1.0
    )
    assert checkpoint_before == json.dumps(
        checkpoint,
        sort_keys=True,
        default=lambda value: value.tolist(),
    )


def test_genome_v2_generation_zero_precedes_segments_and_replays_exactly(
    tmp_path: Path,
) -> None:
    consumer = _consumer(tmp_path)
    generation_zero = _genome_v2_checkpoint("cell-a", generation=0)
    generation_one = _genome_v2_checkpoint("cell-a")
    source_digest = _genome_v2_checkpoint_digest(generation_zero)
    arguments = {
        "cell_ids": ("cell-a",),
        "checkpoint_projections": (generation_zero,),
        "source_projection_sha256_by_cell": {"cell-a": source_digest},
        "source_projection_digest": _genome_v2_checkpoint_digest,
    }

    consumer.observe_generation_zero(**arguments)
    consumer.observe_generation_zero(**arguments)
    with pytest.raises(
        ProviderNativeChronicleError,
        match="replayed Chronicle Genome V2 G0 differs",
    ):
        consumer.observe_generation_zero(
            cell_ids=("cell-a",),
            checkpoint_projections=(generation_zero,),
            source_projection_sha256_by_cell={"cell-a": "c" * 64},
            source_projection_digest=lambda _projection: "c" * 64,
        )
    _observe(
        consumer,
        (_artifact("cell-a"),),
        checkpoint_projections=(generation_one,),
    )
    receipt = consumer.finalize().payload()
    root = Path(receipt["cell_receipts"]["cell-a"]["chronicle_root"])
    state = json.loads(
        (root / "provider_native_chronicle_state.json").read_text()
    )
    report = json.loads((root / ARTIFACT_NAMES["summary"]).read_text())

    assert [
        row["generation"]
        for row in report["genome_v2_generation_summaries"]
    ] == [0, 1]
    assert state["next_segment_index"] == 1
    assert state["event_cursor"] == 1
    assert state["attempt_cursor"] == 1
    assert state["genome_v2_generation_zero_binding"][
        "source_projection_sha256"
    ] == source_digest


def test_genome_v2_generation_zero_rejects_mixed_or_late_authority(
    tmp_path: Path,
) -> None:
    multicell = _consumer(tmp_path / "mixed", ("cell-a", "cell-b"))
    with pytest.raises(
        ProviderNativeChronicleError,
        match="G0 presence differs",
    ):
        multicell.observe_generation_zero(
            cell_ids=("cell-a", "cell-b"),
            checkpoint_projections=(
                _genome_v2_checkpoint("cell-a", generation=0),
                {},
            ),
            source_projection_sha256_by_cell={
                "cell-a": "a" * 64,
                "cell-b": "b" * 64,
            },
            source_projection_digest=_genome_v2_checkpoint_digest,
        )
    assert not (tmp_path / "mixed" / "cells").exists()

    late = _consumer(tmp_path / "late")
    _observe(
        late,
        (_artifact("cell-a"),),
        checkpoint_projections=(_genome_v2_checkpoint("cell-a"),),
    )
    with pytest.raises(
        ProviderNativeChronicleError,
        match="arrived after segment authority",
    ):
        late.observe_generation_zero(
            cell_ids=("cell-a",),
            checkpoint_projections=(
                _genome_v2_checkpoint("cell-a", generation=0),
            ),
            source_projection_sha256_by_cell={
                "cell-a": _genome_v2_checkpoint_digest(
                    _genome_v2_checkpoint("cell-a", generation=0)
                )
            },
            source_projection_digest=_genome_v2_checkpoint_digest,
        )


def test_genome_v2_generation_zero_and_segment_milestones_are_exact(
    tmp_path: Path,
) -> None:
    consumer = _consumer(tmp_path / "g0")
    wrong_generation = _genome_v2_checkpoint("cell-a")
    with pytest.raises(
        ProviderNativeChronicleError,
        match="G0 generation or tick differs",
    ):
        consumer.observe_generation_zero(
            cell_ids=("cell-a",),
            checkpoint_projections=(wrong_generation,),
            source_projection_sha256_by_cell={
                "cell-a": _genome_v2_checkpoint_digest(wrong_generation)
            },
            source_projection_digest=_genome_v2_checkpoint_digest,
        )

    generation_zero = _genome_v2_checkpoint("cell-a", generation=0)
    with pytest.raises(
        ProviderNativeChronicleError,
        match="source projection digest differs",
    ):
        consumer.observe_generation_zero(
            cell_ids=("cell-a",),
            checkpoint_projections=(generation_zero,),
            source_projection_sha256_by_cell={"cell-a": "a" * 64},
            source_projection_digest=_genome_v2_checkpoint_digest,
        )

    with pytest.raises(
        ProviderNativeChronicleError,
        match="segment milestone differs",
    ):
        _observe(
            _consumer(tmp_path / "segment"),
            (_artifact("cell-a"),),
            checkpoint_projections=(
                _genome_v2_checkpoint("cell-a", generation=2),
            ),
        )


def test_checkpoint_rebind_rejects_semantic_genome_v2_summary_reordering(
    tmp_path: Path,
) -> None:
    first = (_artifact("cell-a"),)
    source = _consumer(tmp_path / "source")
    generation_zero = _genome_v2_checkpoint("cell-a", generation=0)
    source.observe_generation_zero(
        cell_ids=("cell-a",),
        checkpoint_projections=(generation_zero,),
        source_projection_sha256_by_cell={
            "cell-a": _genome_v2_checkpoint_digest(generation_zero)
        },
        source_projection_digest=_genome_v2_checkpoint_digest,
    )
    _observe(
        source,
        first,
        checkpoint_projections=(_genome_v2_checkpoint("cell-a"),),
    )
    payload = source.finalize().payload()
    root = Path(payload["cell_receipts"]["cell-a"]["chronicle_root"])
    state_path = root / "provider_native_chronicle_state.json"
    state = json.loads(state_path.read_text())
    state["genome_v2_summaries"].reverse()
    state_path.write_text(json.dumps(state, sort_keys=True))
    payload["cell_receipts"]["cell-a"]["durable_artifact_sha256"][
        "provider_native_chronicle_state.json"
    ] = hashlib.sha256(state_path.read_bytes()).hexdigest()

    with pytest.raises(
        ProviderNativeChronicleError,
        match="summary identity or order differs",
    ):
        ProviderNativeChronicleConsumerV1.from_checkpoint_continuation(
            batch_root=tmp_path / "restored",
            campaign_id="campaign-a",
            batch_id="batch-restored",
            cell_ids=("cell-a",),
            profile="scientific_full",
            prior_receipt=ProviderNativeChronicleReceiptV1.from_payload(
                payload
            ),
            prior_continuity=_continuity(first),
            prior_authority_sha256="a" * 64,
        )


def _continuity(artifacts: tuple[dict, ...]) -> dict:
    return {
        "schema_version": CONTINUITY_SCHEMA_VERSION,
        "segment_count": 1,
        "cell_chain_sha256": [
            artifact["cell"]["chain_sha256_after_segment"]
            for artifact in artifacts
        ],
        "cumulative_attempt_counts": [1 for _ in artifacts],
        "cumulative_success_counts": [1 for _ in artifacts],
        "cumulative_failure_counts": [0 for _ in artifacts],
        "cumulative_snapshot_counts": [2 for _ in artifacts],
    }


def test_checkpoint_rebind_copies_exact_prefix_and_starts_at_next_segment(
    tmp_path: Path,
) -> None:
    cell_ids = ("cell-a", "cell-b")
    first = (
        _artifact("cell-a", path=0),
        _artifact("cell-b", path=1),
    )
    source = _consumer(tmp_path / "source", cell_ids)
    _observe(source, first)
    receipt = source.finalize()
    continuity = _continuity(first)

    restored = ProviderNativeChronicleConsumerV1.from_checkpoint_continuation(
        batch_root=tmp_path / "restored",
        campaign_id="campaign-a",
        batch_id="batch-restored",
        cell_ids=cell_ids,
        profile="scientific_full",
        prior_receipt=ProviderNativeChronicleReceiptV1.from_payload(
            receipt.payload()
        ),
        prior_continuity=continuity,
        prior_authority_sha256="a" * 64,
    )
    second = tuple(
        _artifact(
            cell_id,
            segment_index=1,
            path=index,
            previous_chain_sha256=first[index]["cell"][
                "chain_sha256_after_segment"
            ],
        )
        for index, cell_id in enumerate(cell_ids)
    )
    _observe(restored, second, segment_index=1)
    continued = restored.finalize().payload()

    for cell_id in cell_ids:
        cell_receipt = continued["cell_receipts"][cell_id]
        assert cell_receipt["segment_count"] == 2
        assert cell_receipt["event_cursor"] == 2
        assert cell_receipt["continuation_binding"] == {
            "schema_version": CONTINUATION_REBIND_SCHEMA_VERSION,
            "source_campaign_id": "campaign-a",
            "source_batch_id": "batch-a",
            "restored_batch_id": "batch-restored",
            "prior_authority_sha256": "a" * 64,
            "prior_receipt_sha256": receipt.digest_sha256,
            "prior_continuity_sha256": _digest(continuity),
            "next_segment_index": 1,
            "cell_ids": ["cell-a", "cell-b"],
            "cell_chain_sha256": continuity["cell_chain_sha256"],
        }


def test_checkpoint_rebind_preserves_g0_and_appends_exact_next_summary(
    tmp_path: Path,
) -> None:
    first = (_artifact("cell-a"),)
    source = _consumer(tmp_path / "source")
    generation_zero = _genome_v2_checkpoint("cell-a", generation=0)
    source.observe_generation_zero(
        cell_ids=("cell-a",),
        checkpoint_projections=(generation_zero,),
        source_projection_sha256_by_cell={
            "cell-a": _genome_v2_checkpoint_digest(generation_zero)
        },
        source_projection_digest=_genome_v2_checkpoint_digest,
    )
    _observe(
        source,
        first,
        checkpoint_projections=(_genome_v2_checkpoint("cell-a"),),
    )
    restored = ProviderNativeChronicleConsumerV1.from_checkpoint_continuation(
        batch_root=tmp_path / "restored",
        campaign_id="campaign-a",
        batch_id="batch-restored",
        cell_ids=("cell-a",),
        profile="scientific_full",
        prior_receipt=source.finalize(),
        prior_continuity=_continuity(first),
        prior_authority_sha256="a" * 64,
    )
    second = (
        _artifact(
            "cell-a",
            segment_index=1,
            previous_chain_sha256=first[0]["cell"][
                "chain_sha256_after_segment"
            ],
        ),
    )
    _observe(
        restored,
        second,
        segment_index=1,
        checkpoint_projections=(
            _genome_v2_checkpoint("cell-a", generation=2),
        ),
    )
    receipt = restored.finalize().payload()
    root = Path(receipt["cell_receipts"]["cell-a"]["chronicle_root"])
    report = json.loads((root / ARTIFACT_NAMES["summary"]).read_text())

    assert [
        row["generation"]
        for row in report["genome_v2_generation_summaries"]
    ] == [0, 1, 2]


def test_checkpoint_rebind_accepts_legacy_summary_binding_then_adds_exact(
    tmp_path: Path,
) -> None:
    first = (_artifact("cell-a"),)
    source = _consumer(tmp_path / "source")
    _observe(
        source,
        first,
        checkpoint_projections=(_genome_v2_checkpoint("cell-a"),),
    )
    payload = source.finalize().payload()
    root = Path(payload["cell_receipts"]["cell-a"]["chronicle_root"])
    state_path = root / "provider_native_chronicle_state.json"
    state = json.loads(state_path.read_text())
    legacy_fields = {
        "genome_v2_projection_sha256",
        "genome_v2_generation",
        "genome_v2_tick",
        "genome_v2_sidecar_payload_sha256",
    }
    for field in legacy_fields:
        state["segments"][0].pop(field)
        payload["cell_receipts"]["cell-a"]["segment_bindings"][0].pop(
            field
        )
    state_path.write_text(json.dumps(state, sort_keys=True))
    payload["cell_receipts"]["cell-a"]["durable_artifact_sha256"][
        "provider_native_chronicle_state.json"
    ] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    restored = ProviderNativeChronicleConsumerV1.from_checkpoint_continuation(
        batch_root=tmp_path / "restored",
        campaign_id="campaign-a",
        batch_id="batch-restored",
        cell_ids=("cell-a",),
        profile="scientific_full",
        prior_receipt=ProviderNativeChronicleReceiptV1.from_payload(payload),
        prior_continuity=_continuity(first),
        prior_authority_sha256="a" * 64,
    )
    second = (
        _artifact(
            "cell-a",
            segment_index=1,
            previous_chain_sha256=first[0]["cell"][
                "chain_sha256_after_segment"
            ],
        ),
    )
    _observe(
        restored,
        second,
        segment_index=1,
        checkpoint_projections=(
            _genome_v2_checkpoint("cell-a", generation=2),
        ),
    )
    receipt = restored.finalize().payload()
    bindings = receipt["cell_receipts"]["cell-a"]["segment_bindings"]

    assert "genome_v2_projection_sha256" not in bindings[0]
    assert "genome_v2_projection_sha256" in bindings[1]


def test_checkpoint_receipt_freezes_prefix_without_finalizing_live_owner(
    tmp_path: Path,
) -> None:
    first = _artifact("cell-a")
    source = _consumer(tmp_path / "source")
    _observe(source, (first,))

    prefix = source.checkpoint_receipt(
        snapshot_batch_root=tmp_path / "checkpoint-prefix"
    )
    prefix_root = Path(
        prefix.payload()["cell_receipts"]["cell-a"]["chronicle_root"]
    )
    prefix_state = json.loads(
        (prefix_root / "provider_native_chronicle_state.json").read_text()
    )
    live_state_path = (
        tmp_path
        / "source"
        / "cells"
        / "cell-a"
        / "chronicle"
        / "provider_native_chronicle_state.json"
    )
    live_state = json.loads(live_state_path.read_text())

    assert prefix.finalized is True
    assert prefix_state["finalized"] is True
    assert live_state["finalized"] is False

    second = _artifact(
        "cell-a",
        segment_index=1,
        previous_chain_sha256=first["cell"]["chain_sha256_after_segment"],
    )
    _observe(source, (second,), segment_index=1)
    final = source.finalize()

    assert prefix.cell_receipts["cell-a"]["segment_count"] == 1
    assert final.cell_receipts["cell-a"]["segment_count"] == 2
    assert json.loads(
        (prefix_root / "provider_native_chronicle_state.json").read_text()
    ) == prefix_state


def test_checkpoint_rebind_rejects_chain_substitution_without_target_writes(
    tmp_path: Path,
) -> None:
    first = (_artifact("cell-a"),)
    source = _consumer(tmp_path / "source")
    _observe(source, first)
    receipt = source.finalize()
    continuity = _continuity(first)
    continuity["cell_chain_sha256"] = ["9" * 64]
    target = tmp_path / "restored"

    with pytest.raises(
        ProviderNativeChronicleError,
        match="terminal chain differs",
    ):
        ProviderNativeChronicleConsumerV1.from_checkpoint_continuation(
            batch_root=target,
            campaign_id="campaign-a",
            batch_id="batch-restored",
            cell_ids=("cell-a",),
            profile="scientific_full",
            prior_receipt=receipt,
            prior_continuity=continuity,
            prior_authority_sha256="a" * 64,
        )

    assert not target.exists()


@pytest.mark.parametrize(
    "filename",
    (
        "provider_native_chronicle_state.json",
        "formal_events.jsonl",
    ),
)
def test_checkpoint_rebind_rejects_durable_prefix_tampering(
    tmp_path: Path,
    filename: str,
) -> None:
    first = (_artifact("cell-a"),)
    source = _consumer(tmp_path / "source")
    _observe(source, first)
    receipt = source.finalize()
    root = Path(receipt.cell_receipts["cell-a"]["chronicle_root"])
    authority_path = root / filename
    authority_path.write_bytes(authority_path.read_bytes() + b" ")
    target = tmp_path / "restored"

    with pytest.raises(
        ProviderNativeChronicleError,
        match="durable authority digest differs",
    ):
        ProviderNativeChronicleConsumerV1.from_checkpoint_continuation(
            batch_root=target,
            campaign_id="campaign-a",
            batch_id="batch-restored",
            cell_ids=("cell-a",),
            profile="scientific_full",
            prior_receipt=receipt,
            prior_continuity=_continuity(first),
            prior_authority_sha256="a" * 64,
        )

    assert not target.exists()


def test_checkpoint_rebind_rejects_segment_gap_after_verified_seed(
    tmp_path: Path,
) -> None:
    first = (_artifact("cell-a"),)
    source = _consumer(tmp_path / "source")
    _observe(source, first)
    receipt = source.finalize()
    restored = ProviderNativeChronicleConsumerV1.from_checkpoint_continuation(
        batch_root=tmp_path / "restored",
        campaign_id="campaign-a",
        batch_id="batch-restored",
        cell_ids=("cell-a",),
        profile="scientific_full",
        prior_receipt=receipt,
        prior_continuity=_continuity(first),
        prior_authority_sha256="a" * 64,
    )
    gap = _artifact(
        "cell-a",
        segment_index=2,
        previous_chain_sha256=first[0]["cell"]["chain_sha256_after_segment"],
    )

    with pytest.raises(ProviderNativeChronicleError, match="gap or reordering"):
        _observe(restored, (gap,), segment_index=2)


def test_checkpoint_rebind_rechecks_persisted_prior_authority(
    tmp_path: Path,
) -> None:
    first = (_artifact("cell-a"),)
    source = _consumer(tmp_path / "source")
    _observe(source, first)
    restored = ProviderNativeChronicleConsumerV1.from_checkpoint_continuation(
        batch_root=tmp_path / "restored",
        campaign_id="campaign-a",
        batch_id="batch-restored",
        cell_ids=("cell-a",),
        profile="scientific_full",
        prior_receipt=source.finalize(),
        prior_continuity=_continuity(first),
        prior_authority_sha256="a" * 64,
    )
    state_path = (
        tmp_path
        / "restored"
        / "cells"
        / "cell-a"
        / "chronicle"
        / "provider_native_chronicle_state.json"
    )
    state = json.loads(state_path.read_text())
    state["continuation_binding"]["prior_authority_sha256"] = "9" * 64
    state_path.write_text(json.dumps(state))
    second = _artifact(
        "cell-a",
        segment_index=1,
        previous_chain_sha256=first[0]["cell"]["chain_sha256_after_segment"],
    )

    with pytest.raises(
        ProviderNativeChronicleError,
        match="authority or digest differs",
    ):
        _observe(restored, (second,), segment_index=1)


def _plan() -> dict:
    return {
        "campaign_id": "campaign-a",
        "output": {"chronicle_profile": "scientific_full"},
        "execution": {"generations": 1},
        "cells": [{"cell_id": "cell-a"}],
    }


def test_closeout_accepts_generic_validated_provider_native_receipt(
    tmp_path: Path,
) -> None:
    batch_root = tmp_path / "batches" / "batch-a"
    consumer = _consumer(batch_root)
    _observe(consumer, (_artifact("cell-a"),))
    receipt = consumer.finalize().payload()
    manifest = {
        "batch_id": "batch-a",
        "cell_ids": ["cell-a"],
        "status": "completed",
        "provider_id": PRODUCTION_PROVIDER_ID,
        "result": {
            "provider_native_chronicle_required": True,
            "provider_native_chronicle": receipt,
        },
    }

    roots = materialize_completed_campaign_chronicles(
        _plan(), run_root=tmp_path, batch_manifests=[manifest]
    )
    assert roots["cell-a"] == (
        batch_root / "cells" / "cell-a" / "chronicle"
    ).resolve()


def test_closeout_refuses_metrics_fallback_when_native_receipt_is_required(
    tmp_path: Path,
) -> None:
    manifest = {
        "batch_id": "batch-a",
        "cell_ids": ["cell-a"],
        "status": "completed",
        "provider_id": PRODUCTION_PROVIDER_ID,
        "metrics_rows": [
            {"cell_id": "cell-a", "generation": 1, "metrics": {"alive": 2}}
        ],
        "result": {"provider_native_chronicle_required": True},
    }
    with pytest.raises(
        CampaignChronicleCloseoutError,
        match="provider-native Chronicle receipt",
    ):
        materialize_completed_campaign_chronicles(
            _plan(), run_root=tmp_path, batch_manifests=[manifest]
        )
