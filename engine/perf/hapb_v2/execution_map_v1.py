"""Source-grounded HAPB V1 execution anatomy for the V2 prep packet."""

from __future__ import annotations

from typing import Final


EXECUTION_MAP_SCHEMA_V1: Final[str] = "eiel_hapb_v2_execution_map_v1"


def current_execution_map_v1() -> dict[str, object]:
    return {
        "schema_version": EXECUTION_MAP_SCHEMA_V1,
        "base_commit": "2f87dabee57fc767ccbb49ace1078ab3556448c7",
        "active_gate6_branch": "codex/hapb-capacity-repair-20260815",
        "host_canonical_authority": True,
        "stages": [
            {
                "name": "policy_row_storage",
                "cpu_or_gpu": "CPU canonical / GPU reconstructible caches",
                "mutable": "host bank mutable; CUDA caches non-authoritative",
                "source": "engine/perf/cuda_architecture_b_hapb_v1.py:HostAuthoritativePolicyBankV1",
                "inputs": ["canonical_host_bank [48,512,9991] float32"],
                "outputs": ["hot_prefix_cache", "cuda_tile_arena"],
                "serialization_necessary": True,
                "notes": "host is the sole policy authority",
            },
            {
                "name": "content_ids_alias_state",
                "cpu_or_gpu": "CPU",
                "mutable": True,
                "source": "engine/perf/cuda_architecture_b_hapb_v1.py:_content_row_map_from_canonical_v1",
                "inputs": ["canonical_host_bank"],
                "outputs": ["content_row_map", "content_row_versions", "slot_generations"],
                "serialization_necessary": True,
                "notes": (
                    "overflow binds only on exact same-cell prefix digest; "
                    "ordinary birth with alias on raises if child bytes are "
                    "not an exact parent clone; turnover leaves unique "
                    "children unbound and tiles them; no API inserts a new "
                    "identity into the P256 pool"
                ),
            },
            {
                "name": "observations_layout",
                "cpu_or_gpu": "GPU",
                "mutable": True,
                "source": "hapb_*_policy_logits_v1 observations [rows,83] float32",
                "inputs": ["per-row observation"],
                "outputs": ["consumed by exact helper"],
                "serialization_necessary": False,
            },
            {
                "name": "policy_mlp_kernel",
                "cpu_or_gpu": "GPU",
                "mutable": False,
                "source": "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu:fixed_mlp_policy_logits_from_pack_exact_v1",
                "inputs": ["pack", "observation", "slot scalars"],
                "outputs": ["logits [rows,7]"],
                "serialization_necessary": False,
                "launch": "<<<rows,1>>>",
                "useful_threads": 1,
                "thread0_guard": True,
                "confirmed_nx1": True,
            },
            {
                "name": "gumbel_settlement",
                "cpu_or_gpu": "GPU",
                "mutable": True,
                "source": "hapb_settle_prepared_policy_rows_kernel_v1<<<cells,1>>>",
                "inputs": ["logits", "pcg64 state"],
                "outputs": ["actions"],
                "serialization_necessary": True,
                "notes": (
                    "device-side serial Gumbel via pcg64_numpy_float32, "
                    "clip 1e-6, -logf(-logf(u)); not curand; V2 must not "
                    "change RNG semantics"
                ),
            },
            {
                "name": "host_classification_content_map",
                "cpu_or_gpu": "CPU",
                "mutable": True,
                "source": "cuda_architecture_b_hapb_v1.py compact_logits / refill tiles",
                "inputs": ["global row slots", "content_row_map"],
                "outputs": ["gathered tile or alias content rows"],
                "serialization_necessary": "implementation-induced for overflow tiles",
            },
            {
                "name": "h2d_d2h_and_sync",
                "cpu_or_gpu": "CPU+GPU",
                "mutable": False,
                "source": "HostAuthoritativePolicyBankV1._iter_refilled_overflow_tiles_v1",
                "inputs": ["canonical rows"],
                "outputs": ["cuda_tile_arena"],
                "serialization_necessary": "implementation-induced torch.cuda.synchronize after each tile",
            },
            {
                "name": "tile_refill_lifecycle",
                "cpu_or_gpu": "CPU stages into one CUDA tile",
                "mutable": True,
                "source": "cuda_architecture_b_hapb_v1.py tile_rows default 512",
                "inputs": ["overflow global rows"],
                "outputs": ["one retained CUDA tile"],
                "serialization_necessary": True,
                "notes": "4096-row tile was rejected as ~2.3% slower",
            },
            {
                "name": "birth_mutation_publication",
                "cpu_or_gpu": "host commit then optional alias bind",
                "mutable": True,
                "source": "bind_content_alias_v1; hapb_ordinary_birth_mutate_rows_v1",
                "inputs": ["parent bytes", "child bytes"],
                "outputs": [
                    "host bank updated first",
                    "ordinary-birth alias-on: exact clone only or raise",
                    "turnover: unique children stay unbound and tiled",
                ],
                "serialization_necessary": True,
            },
            {
                "name": "checkpoint_restore",
                "cpu_or_gpu": "CPU authority; CUDA caches excluded",
                "mutable": False,
                "source": "export_hapb_policy_bank_checkpoint_v1 schema v2",
                "inputs": ["canonical bank + compact alias state"],
                "outputs": ["fresh owner must rehydrate caches"],
                "serialization_necessary": True,
            },
            {
                "name": "turnover_slot_aba",
                "cpu_or_gpu": "CPU generations + CUDA identity pairs",
                "mutable": True,
                "source": "slot_generations; active identity pairs",
                "inputs": ["slot reuse events"],
                "outputs": ["generation bump / ABA"],
                "serialization_necessary": True,
            },
        ],
        "verified_findings": {
            "host_canonical_authority": True,
            "nx1_thread0_only": True,
            "gumbel_device_side": True,
            "cross_cell_sharing_implemented": False,
            "alias_new_unique_overflow": (
                "ordinary_birth_raises; turnover_unbound_tiled; no_pool_insert"
            ),
            "preferred_tile_rows": 512,
        },
    }
