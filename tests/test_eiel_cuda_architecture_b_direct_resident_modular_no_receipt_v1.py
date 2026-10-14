from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from engine.perf import cuda_architecture_b_direct_resident_modular_v1 as modular
from perf.cuda_architecture_b_modular_state_abi import (
    pack_modular_storage,
    rebind_modular_lifecycle_storage,
)
from tests.perf.test_eiel_cuda_architecture_b_modular_state_abi import (
    _module,
    _record,
)
from tests.test_eiel_cuda_architecture_b_direct_resident_modular_v1 import (
    _cell_tensors,
    _lifecycle_capability,
    _patch,
)


MUTATION_SIDECAR_NAMES = (
    "turnover_mutation_draws",
    "duplicate_descendant_mask",
    "duplicate_descendant_mask_shadow",
    "turnover_mutation_receipt_rows",
    "turnover_mutation_receipt_counts",
    "turnover_mutation_receipt_overflow",
)


def test_qualification_disabled_preparation_has_no_receipt_burden() -> None:
    captured: dict[str, Any] = {}

    class Module:
        @staticmethod
        def eiel_cuda_architecture_b_flattened_modular_lifecycle_capability_v2():
            return _lifecycle_capability()

        @staticmethod
        def eiel_cuda_architecture_b_execute_direct_resident_modular_lifecycle_physical_v2(
            tensors: list[torch.Tensor],
            config: dict[str, Any],
            *_arguments: Any,
        ) -> dict[str, Any]:
            captured.update(tensors=tensors, config=config)
            return {
                "modular_native_owner": True,
                "modular_native_controller_abi_version": 2,
                "modular_physical_gate_entrypoint": True,
                "modular_python_tick_or_generation_ownership": False,
                "modular_lifecycle_supported": True,
                "hidden_fallback": False,
                "runtime_jit": False,
            }

    prepared = modular.prepare_direct_resident_modular_lifecycle_v2(
        _patch(), [_cell_tensors()]
    )

    assert all(getattr(prepared, name) is None for name in MUTATION_SIDECAR_NAMES)
    assert modular.project_modular_turnover_mutation_receipt_v1(prepared) is None
    assert modular.project_modular_turnover_mutation_sequence_v1(prepared) is None

    modular.execute_direct_resident_modular_lifecycle_physical_v2(
        Module(),
        prepared,
        generation_count=1,
        ticks_per_generation=1,
    )

    assert set(MUTATION_SIDECAR_NAMES).isdisjoint(captured["config"])
    assert "turnover_mutation_first_generation" not in captured["config"]


def test_no_receipt_retains_exact_parent_copy_semantics() -> None:
    prepared = modular.prepare_direct_resident_modular_lifecycle_v2(
        _patch(), [_cell_tensors()]
    )
    mutation_receipt = modular.project_modular_turnover_mutation_receipt_v1(prepared)
    parent = _record(
        [_module(0), _module(1, gated=True)],
        with_runtime=True,
        agent_id=10,
        agent_slot_token=100,
    )
    authority = pack_modular_storage(
        [parent, None, None, None],
        agent_capacity=4,
        profile_id="b2-12x32-v1",
    )
    expected_child = deepcopy(parent)
    expected_child["agent_id"] = 20
    expected_child["agent_slot_token"] = 200
    for name in (
        "structural_live_module_state",
        "structural_live_module_outputs",
        "structural_development_latch_by_instance_id",
        "structural_development_effective_gain_by_instance_id",
    ):
        expected_child[name] = {}
    expected = pack_modular_storage(
        [parent, expected_child, None, None],
        agent_capacity=4,
        profile_id="b2-12x32-v1",
    )

    rebound = rebind_modular_lifecycle_storage(
        authority,
        profile_id="b2-12x32-v1",
        terminal_alive_mask=torch.tensor([True, True, False, False]),
        terminal_slot_to_agent_id=torch.tensor([10, 20, -1, -1]),
        terminal_slot_tokens=torch.tensor([100, 200, 0, 0]),
        selected_parent_slot_by_child_slot={1: 0},
        mutation_receipt=mutation_receipt,
    )

    assert mutation_receipt is None
    assert rebound.canonical_bytes() == expected.canonical_bytes()
