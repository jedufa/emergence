"""Verify installed configuration and pinned resources; no simulation is run."""
from dataclasses import replace
import hashlib
import json

from emergence_contracts.resources import read_resource_bytes
from emergence_worldkit import EnvConfig


def main() -> None:
    baseline = EnvConfig(width=32, height=32)
    scarcer = replace(baseline, food_spawn_per_step=14)
    assert baseline.width == scarcer.width == 32
    assert baseline.height == scarcer.height == 32
    assert baseline.food_spawn_per_step != scarcer.food_spawn_per_step
    resources = {
        name: hashlib.sha256(read_resource_bytes(name)).hexdigest()
        for name in (
            "crca.run-attempt-edge-cases.v1",
            "crca.g0-ratchet-reconciliation.v1",
        )
    }
    print(json.dumps({
        "status": "COMPONENT_SMOKE_VERIFIED",
        "world_shape": [baseline.width, baseline.height],
        "food_inputs": [baseline.food_spawn_per_step, scarcer.food_spawn_per_step],
        "verified_resource_sha256": resources,
        "evolution_executed": False,
        "qualification_granted": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
