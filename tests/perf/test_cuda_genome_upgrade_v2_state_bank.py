from __future__ import annotations

import pytest
import torch

from evolution.genome_upgrade_v2 import (
    CONTEXT_COUNT,
    PRODUCT_COUNT,
    Q10_SCALE,
    GenomeUpgradeV2V1,
    express_reference_v1,
)
from evolution.genome_upgrade_v2_telemetry import (
    summarize_genome_upgrade_v2_resident_tensors,
)
from perf.cuda_genome_upgrade_v2_state_bank import (
    CudaGenomeUpgradeV2StateBank,
    GENOME_V2_FOUNDER_PROFILE_SCHEMA,
    resolve_genome_v2_founder_profile_v1,
    restore_cuda_genome_upgrade_v2_state_bank,
)
from evolution.genome_upgrade_v2_capability_contract_v1 import (
    GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256,
)


def _nonneutral() -> GenomeUpgradeV2V1:
    value = GenomeUpgradeV2V1.neutral()
    alleles = [
        [list(parameters) for parameters in homolog] for homolog in value.alleles
    ]
    modifiers = [list(homolog) for homolog in value.modifiers]
    alleles[0][4] = [700, 800, 900, -300]
    alleles[1][4] = [-200, 600, 700, 400]
    modifiers[0][4] = 250
    modifiers[1][4] = -100
    return GenomeUpgradeV2V1(
        alleles=tuple(
            tuple(tuple(parameters) for parameters in homolog) for homolog in alleles
        ),
        modifiers=tuple(tuple(homolog) for homolog in modifiers),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fixed_state_bank_expression_and_checkpoint(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    bank = CudaGenomeUpgradeV2StateBank.allocate(cells=2, capacity=3, device=device)
    assert bank.alleles_q10.shape == (2, 2, PRODUCT_COUNT, 4, 3)
    value = _nonneutral()
    bank.set_genotype(1, 2, value)
    assert bank.genotype(1, 2) == value
    context = torch.zeros((2, 3, CONTEXT_COUNT), dtype=torch.int64, device=device)
    context[1, 2, 1] = 512
    pointers = {
        name: tensor.data_ptr() for name, tensor in bank.named_tensors().items()
    }
    bank.express(context)
    assert pointers == {
        name: tensor.data_ptr() for name, tensor in bank.named_tensors().items()
    }
    expected = express_reference_v1(value, (0, 512, 0, 0, 0, 0, 0, 0))
    assert tuple(bank.products_q10[1, :, 2].cpu().tolist()) == (expected.products_q10)
    assert tuple(bank.phenotypes_q10[1, :, 2].cpu().tolist()) == (
        expected.phenotypes_q10
    )
    assert int(bank.expression_cost_q10[1, 2]) == expected.expression_cost_q10

    restored = restore_cuda_genome_upgrade_v2_state_bank(
        bank.checkpoint_projection(), device=device
    )
    assert (
        bank.checkpoint_projection()["capability_contract_sha256"]
        == GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256
    )
    for name, tensor in bank.named_tensors().items():
        assert torch.equal(tensor, restored.named_tensors()[name])


def test_checkpoint_rejects_stale_capability_contract() -> None:
    bank = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=1, device="cpu")
    payload = bank.checkpoint_projection()
    payload["capability_contract_sha256"] = "0" * 64
    with pytest.raises(Exception, match="capability contract hash mismatch"):
        restore_cuda_genome_upgrade_v2_state_bank(payload, device="cpu")


def test_one_cpu_snapshot_drives_projection_without_aliasing_resident_bank() -> None:
    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=1,
        capacity=2,
        device="cpu",
    )
    bank.alleles_q10[0, 0, 0, 0, 0] = 123

    snapshot = bank.checkpoint_cpu_tensors_v1()
    projection = bank.checkpoint_projection(cpu_tensors=snapshot)
    bank.alleles_q10[0, 0, 0, 0, 0] = 456

    assert snapshot["genome_v2_alleles_q10"][0, 0, 0, 0, 0].item() == 123
    assert projection["tensors"]["genome_v2_alleles_q10"][0][0][0][0][0] == 123


def test_bank_storage_is_bounded_and_linear() -> None:
    small = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=1, device="cpu")
    large = CudaGenomeUpgradeV2StateBank.allocate(cells=4, capacity=8, device="cpu")
    assert large.storage_bytes() == small.storage_bytes() * 32
    # Current exact int64 envelope is 1,160 bytes per resident agent.
    assert small.storage_bytes() == 1160
    assert small.storage_bytes() < 2048


def test_neutral_bank_uses_homozygous_neutral_quantity_and_stability() -> None:
    bank = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=2, device="cpu")
    assert torch.count_nonzero(bank.alleles_q10[:, :, :, 0, :]) == 0
    assert torch.all(bank.alleles_q10[:, :, :, 1, :] == Q10_SCALE)
    assert torch.all(bank.alleles_q10[:, :, :, 2, :] == Q10_SCALE)
    assert torch.count_nonzero(bank.alleles_q10[:, :, :, 3, :]) == 0


@pytest.mark.parametrize(
    "profile_id",
    [
        "neutral",
        "pleiotropic_tradeoffs",
        "regulatory_activation_inhibition",
    ],
)
def test_frozen_founder_profiles_initialize_live_slots_only(
    profile_id: str,
) -> None:
    profile = resolve_genome_v2_founder_profile_v1(profile_id)
    bank = CudaGenomeUpgradeV2StateBank.allocate(cells=2, capacity=3, device="cpu")
    neutral = bank.genotype(0, 2)
    alive = torch.tensor([[True, False, False], [False, True, False]], dtype=torch.bool)
    bank.initialize_founders_v1(tuple(profile["genotype_q10"]), alive_mask=alive)

    assert bank.genotype(0, 0) == bank.genotype(1, 1)
    assert bank.genotype(0, 2) == neutral
    if profile_id == "neutral":
        assert bank.genotype(0, 0) == neutral
    else:
        assert bank.genotype(0, 0) != neutral


def test_checkpoint_founder_profile_authority_is_exact() -> None:
    profile = resolve_genome_v2_founder_profile_v1("regulatory_activation_inhibition")
    identity = {
        "schema_version": GENOME_V2_FOUNDER_PROFILE_SCHEMA,
        "profile_id": profile["profile_id"],
        "payload_sha256": profile["payload_sha256"],
    }
    bank = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=1, device="cpu")
    payload = {**bank.checkpoint_projection(), "founder_profile": identity}
    restore_cuda_genome_upgrade_v2_state_bank(
        payload,
        device="cpu",
        expected_founder_profile=profile,
    )
    payload["founder_profile"] = {**identity, "profile_id": "neutral"}
    with pytest.raises(Exception, match="founder profile authority mismatch"):
        restore_cuda_genome_upgrade_v2_state_bank(
            payload,
            device="cpu",
            expected_founder_profile=profile,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_b2_founder_initialization_is_per_cell_and_ordered(
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    neutral = resolve_genome_v2_founder_profile_v1("neutral")
    treatment = resolve_genome_v2_founder_profile_v1("pleiotropic_tradeoffs")
    bank = CudaGenomeUpgradeV2StateBank.allocate(cells=2, capacity=3, device=device)
    bank.initialize_founders_by_cell_v1(
        (
            tuple(neutral["genotype_q10"]),
            tuple(treatment["genotype_q10"]),
        ),
        alive_mask=torch.tensor(
            [[True, True, False], [True, True, False]],
            dtype=torch.bool,
            device=device,
        ),
    )

    assert bank.genotype(0, 0) == GenomeUpgradeV2V1.neutral()
    assert bank.genotype(1, 0) != GenomeUpgradeV2V1.neutral()
    assert bank.genotype(0, 2) == bank.genotype(1, 2)


def test_compact_summary_is_bounded_and_alive_only() -> None:
    bank = CudaGenomeUpgradeV2StateBank.allocate(cells=2, capacity=3, device="cpu")
    bank.set_genotype(0, 1, _nonneutral())
    contexts = torch.zeros((2, 3, CONTEXT_COUNT), dtype=torch.int64)
    contexts[0, 1, 5] = Q10_SCALE
    bank.express(contexts)
    summary = summarize_genome_upgrade_v2_resident_tensors(
        bank.named_tensors(),
        alive_mask=torch.tensor(
            [[False, True, False], [True, True, False]], dtype=torch.bool
        ),
        context_q10=contexts,
        consequence_rows=(
            {"cell": 0, "surface": "movement", "applied_delta": -2.0},
            {
                "cell": 0,
                "surface": "movement",
                "applied_delta": -1.0,
                "cap_bound": True,
            },
        ),
        selected_events=tuple({"cell": 0, "event": index} for index in range(20)),
    )
    assert summary["raw_tick_arrays_retained"] is False
    assert summary["cells"][0]["population_count"] == 1
    assert summary["cells"][1]["population_count"] == 2
    assert summary["cells"][0]["contexts"]["channels"]["5"]["exposure_frequency"] == 1.0
    consequences = summary["cells"][0]["consequences"]
    assert consequences["applied_delta_by_surface"]["movement"] == -3.0
    assert consequences["cap_binding_count_by_surface"]["movement"] == 1
    assert summary["cells"][0]["expression_cost_total_q10"] >= 0
    decomposition = summary["cells"][0]["polygenic_decomposition"]
    assert len(decomposition["phenotypes"]) == 10
    assert all(
        len(row["top_product_indices"]) == 3 for row in decomposition["phenotypes"]
    )
    assert (
        summary["cells"][0]["products_q10"]["energy_flux"]["saturation_frequency"]
        is not None
    )
    assert (
        summary["cells"][0]["phenotypes_q10"]["body_size"]["saturation_frequency"]
        is not None
    )
    hereditary = summary["cells"][0]["hereditary_capacity"]
    assert hereditary["population_count"] == 1
    assert hereditary["allele_parameter_heterozygous_count"] == 4
    assert hereditary["modifier_heterozygous_count"] == 1
    assert hereditary["distinct_genotype_count"] == 1
    assert hereditary["hidden_genotype_retained_agent_count"] == 1
    assert hereditary["locus_summary_count"] == 60
    assert hereditary["raw_genotype_rows_retained"] is False
    assert len(summary["cells"][0]["selected_causal_events"]) == 16
