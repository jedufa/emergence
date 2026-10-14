from __future__ import annotations

from dataclasses import replace

import pytest

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.contracts import (
    CampaignPlanV1,
    LogicalCellV1,
    PhysicalBatchV1,
)
from engine.campaigns.genome_v2_recomb72_seed_authority_v1 import (
    DESIGN_LABEL,
    ECOLOGY_SEEDS,
    GenomeV2Recomb72SeedAuthorityError,
    audit_genome_v2_recomb72_seed_authority_v1,
)
from engine.evolution.genome_upgrade_v2_recomb72_founders_v1 import (
    FOUNDER_PROFILE_IDS,
)


HASH = "a" * 64


def _cell(profile: str, ecology: int, ordinal: int) -> LogicalCellV1:
    identity = canonical_sha256(
        {
            "profile": profile,
            "ecology": ecology,
            "task": 0,
        }
    )
    return LogicalCellV1(
        ordinal=ordinal,
        cell_id=f"cell-{identity[:20]}",
        branch_id=f"branch-{identity[20:40]}",
        rng_lineage_id=f"rng-{identity[40:60]}",
        dimensions={
            "ecology_seed": ecology,
            "founder_profile": profile,
            "task_seed": 0,
        },
        parameters={"genome_upgrade_v2.founder_profile": profile},
        seeds={"ecology": ecology, "task": 0},
    )


def _plan() -> CampaignPlanV1:
    cells = tuple(
        _cell(profile, ecology, ordinal)
        for ordinal, (ecology, profile) in enumerate(
            (ecology, profile)
            for ecology in ECOLOGY_SEEDS
            for profile in FOUNDER_PROFILE_IDS
        )
    )
    plan = CampaignPlanV1(
        schema_version="eiel_campaign_plan_v1",
        campaign_id="genome-v2-recomb72",
        description="test",
        no_claims="bounded seed authority only",
        profile_id="eiel-production-runtime-v1",
        qualification_key={},
        qualification_key_sha256=HASH,
        generations=30,
        ticks_per_generation=80,
        checkpoints=(0, 30),
        digest_every_generation=True,
        checkpoint_storage_mode="disk",
        budgets={},
        cells=cells,
        batches=(
            PhysicalBatchV1(
                ordinal=0,
                batch_id="batch-0000",
                cell_ids=tuple(cell.cell_id for cell in cells),
            ),
        ),
        comparison_branches=(),
        analysis={},
        artifacts={},
        output={},
        resource_estimate={},
    )
    return replace(
        plan,
        plan_sha256=canonical_sha256(plan.payload(include_hash=False)),
    )


def _audit(plan: CampaignPlanV1) -> dict:
    return audit_genome_v2_recomb72_seed_authority_v1(
        plan,
        package_id_sha256="1" * 64,
        execution_route_sha256="2" * 64,
        world_identity="patch",
        configuration_sha256="3" * 64,
    )


def _rehash(plan: CampaignPlanV1, **changes: object) -> CampaignPlanV1:
    changed = replace(plan, **changes, plan_sha256="")
    return replace(
        changed,
        plan_sha256=canonical_sha256(changed.payload(include_hash=False)),
    )


def test_audit_freezes_ecology_blocked_independent_selection_design() -> None:
    result = _audit(_plan())

    assert result["design_label"] == DESIGN_LABEL
    assert result["fully_paired_claims_allowed"] is False
    assert "Fully paired trajectory claims are prohibited" in result[
        "claims_boundary"
    ]
    assert len(result["rows"]) == 72
    assert len({row["master_seed_i63"] for row in result["rows"]}) == 6
    assert {
        sum(
            candidate["master_seed_i63"] == row["master_seed_i63"]
            for candidate in result["rows"]
        )
        for row in result["rows"]
    } == {12}
    assert len(
        {
            row["turnover_selection_seed_u64"]
            for row in result["rows"]
        }
    ) == 72
    assert all(result["checks"].values())
    unsigned = dict(result)
    unsigned.pop("audit_sha256")
    assert result["audit_sha256"] == canonical_sha256(unsigned)


def test_audit_is_deterministic_and_cell_declaration_order_invariant() -> None:
    plan = _plan()
    first = _audit(plan)
    assert _audit(plan) == first

    reordered = _rehash(plan, cells=tuple(reversed(plan.cells)))
    second = _audit(reordered)
    assert second["rows"] == first["rows"]
    assert second["rows_sha256"] == first["rows_sha256"]
    assert (
        second["physical_batch_identity_mapping_sha256"]
        == first["physical_batch_identity_mapping_sha256"]
    )


def test_turnover_seed_vector_follows_identity_within_ecology_blocks() -> None:
    plan = _plan()
    permuted_ids = list(reversed(plan.batches[0].cell_ids))
    permuted = _rehash(
        plan,
        batches=(
            replace(plan.batches[0], cell_ids=tuple(permuted_ids)),
        ),
    )

    original = _audit(plan)
    changed = _audit(permuted)
    original_by_id = {
        row["cell_id"]: row["turnover_selection_seed_u64"]
        for row in original["rows"]
    }
    assert changed["rows_sha256"] == original["rows_sha256"]
    assert [
        (row["cell_id"], row["turnover_selection_seed_u64"])
        for row in changed["physical_batch_mapping"]
    ] == [
        (cell_id, original_by_id[cell_id]) for cell_id in permuted_ids
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing_cell", "exactly 72 cells"),
        ("second_batch", "exactly one physical B72"),
        ("duplicate_cell", "every founder-profile/ecology pair exactly once"),
    ],
)
def test_audit_fails_closed_on_matrix_or_physical_design_drift(
    change: str,
    message: str,
) -> None:
    plan = _plan()
    if change == "missing_cell":
        changed = _rehash(
            plan,
            cells=plan.cells[:-1],
            batches=(
                replace(plan.batches[0], cell_ids=plan.batches[0].cell_ids[:-1]),
            ),
        )
    elif change == "second_batch":
        changed = _rehash(
            plan,
            batches=(
                replace(plan.batches[0], cell_ids=plan.batches[0].cell_ids[:36]),
                PhysicalBatchV1(
                    ordinal=1,
                    batch_id="batch-0001",
                    cell_ids=plan.batches[0].cell_ids[36:],
                ),
            ),
        )
    else:
        duplicate = replace(
            plan.cells[-1],
            parameters=plan.cells[0].parameters,
            dimensions=plan.cells[0].dimensions,
        )
        changed = _rehash(
            plan,
            cells=(*plan.cells[:-1], duplicate),
        )

    with pytest.raises(GenomeV2Recomb72SeedAuthorityError, match=message):
        _audit(changed)


def test_audit_fails_closed_on_plan_or_authority_identity_drift() -> None:
    plan = replace(_plan(), plan_sha256="0" * 64)
    with pytest.raises(
        GenomeV2Recomb72SeedAuthorityError,
        match="does not bind",
    ):
        _audit(plan)

    with pytest.raises(
        GenomeV2Recomb72SeedAuthorityError,
        match="package_id_sha256",
    ):
        audit_genome_v2_recomb72_seed_authority_v1(
            _plan(),
            package_id_sha256="not-a-hash",
            execution_route_sha256="2" * 64,
            world_identity="patch",
            configuration_sha256="3" * 64,
        )
