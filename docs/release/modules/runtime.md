# Runtime

Existing `engine.native_runtime`, `engine.authoritative_state`, `engine.simulation`, and fused `engine.perf` implementations remain authoritative.

Example: Use the Composition science front door and authenticated checkpoint occurrences. Read `engine/native_runtime/cuda_snapshot_v1/format.py` for the exact format.

Runtime owns resident state, RNG cursors and restore. Contributors may improve packaging and observability without altering execution; native changes require exact tick/RNG/checkpoint parity.

Dependencies follow [the source map](../source-to-module.json) and the existing ten-module DAG. Shared legacy imports remain under the unchanged debt ratchet.

Important checks: `tests/tick_native_generation_firewall.py`, native source-bound manifests and targeted checkpoint parity.

Status: Native/GPU execution is an advanced route; no empty `emergence_runtime` facade was added.
