from __future__ import annotations

import os

import pytest
import torch

from engine.tw_rngv2.address import EventAddress
from engine.tw_rngv2.contract import NAMESPACE_DEFAULT, SEMANTIC_EPOCH
from engine.tw_rngv2.cuda import load_cuda_extension, philox_device
from engine.tw_rngv2.oracle import PrototypeOracle
from engine.tw_rngv2.philox import PHILOX_KATS


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("EIEL_TWRNGV2_GPU") != "1",
    reason="exclusive GPU admission not granted for TW-RNGV2 CUDA tests",
)


@cuda_required
def test_official_philox_kats_on_cuda() -> None:
    module = load_cuda_extension()
    counters = torch.tensor(
        [list(row[1]) for row in PHILOX_KATS if row[0] == 10],
        dtype=torch.uint32,
        device="cuda",
    )
    keys = torch.tensor(
        [list(row[2]) for row in PHILOX_KATS if row[0] == 10],
        dtype=torch.uint32,
        device="cuda",
    )
    expected = torch.tensor(
        [list(row[3]) for row in PHILOX_KATS if row[0] == 10],
        dtype=torch.uint32,
        device="cuda",
    )
    got = philox_device(counters, keys, rounds=10, module=module)
    assert torch.equal(got, expected)


@cuda_required
def test_cpu_gpu_parity_sample() -> None:
    module = load_cuda_extension()
    oracle = PrototypeOracle()
    addresses = [
        EventAddress(world, 0x0331, subject + 1, 4, 2)
        for world, subject in ((0, 0), (1, 7), (55, 99), (63, 1023))
    ]
    cpu = [oracle.generate(address, n=7) for address in addresses]
    world = torch.tensor([a.world_uid for a in addresses], dtype=torch.uint32, device="cuda")
    domain = torch.tensor([a.domain_id for a in addresses], dtype=torch.uint32, device="cuda")
    epoch = torch.tensor(
        [SEMANTIC_EPOCH] * len(addresses), dtype=torch.uint32, device="cuda"
    )
    subject = torch.tensor(
        [a.subject_uid for a in addresses], dtype=torch.uint64, device="cuda"
    )
    tick = torch.tensor(
        [a.absolute_tick for a in addresses], dtype=torch.uint64, device="cuda"
    )
    operation = torch.tensor(
        [a.operation_id for a in addresses], dtype=torch.uint32, device="cuda"
    )
    n_values = torch.tensor([7] * len(addresses), dtype=torch.uint64, device="cuda")
    outputs = module.generate(
        world, domain, epoch, subject, tick, operation, n_values, NAMESPACE_DEFAULT
    )
    w0, w1, w2, w3, bounded, accepted, _, _, z0, z1, _, status = outputs
    z0_host = z0.detach().cpu().numpy()
    z1_host = z1.detach().cpu().numpy()
    for index, result in enumerate(cpu):
        assert int(status[index]) == int(result.status)
        assert int(w0[index]) == result.words[0]
        assert int(w1[index]) == result.words[1]
        assert int(w2[index]) == result.words[2]
        assert int(w3[index]) == result.words[3]
        assert int(bounded[index]) == result.bounded
        assert int(accepted[index]) == result.accepted_ordinal
        assert int(z0_host[index]) == result.z0_bits
        assert int(z1_host[index]) == result.z1_bits
