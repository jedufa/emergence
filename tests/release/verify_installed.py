"""Verify installed component closure, without claiming evolutionary execution.

Run from outside the checkout with the clean environment's Python -I and this standalone script.
The request is read-only; this command creates no simulation or runtime state.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import sys


MODULES = (
    "engine", "emergence_contracts.resources", "emergence_worldkit",
    "emergence_agentkit", "emergence_lab", "emergence_evidence.science_compact_world_atlas_v1",
    "eiel_composition.science_request_v1_front_door",
    "infra.webui.server_chronicle", "tools.qualification_review.engine",
    "tools.evidence.librarian.cli", "tools.evidence.librarian.publication",
    "tools.evidence.librarian.ingress", "scripts.utils.run_evo",
    "experiments.harness.run_experiments",
    "tools.visualization.living_lineage_canopy.python.provenance",
    "tools.visualization.living_lineage_canopy.python.schema",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated or sys.prefix == sys.base_prefix:
        parser.error("use an isolated (-I) clean virtual environment")
    prefix = Path(sys.prefix).resolve()
    origins = {}
    for name in MODULES:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(prefix), (name, path)
        origins[name] = str(path)

    import numpy as np
    import torch
    from emergence_worldkit import EnvConfig
    from env.env_config import EnvConfig as LegacyEnvConfig
    from env.gridworld_pkg.world import GridWorld
    from emergence_agentkit import MLPPolicy, clone_parameters
    from agents.agent_policy import MLPPolicy as LegacyMLPPolicy
    from emergence_lab import RegistryAxis
    from experiments.harness.schema import RegistryAxis as LegacyRegistryAxis
    from emergence_contracts.resources import read_resource_bytes
    from eiel_composition.science_request_v1_front_door import validate_science_request_v1

    assert EnvConfig is LegacyEnvConfig
    preset = replace(EnvConfig(), width=4, height=4, patch_world_enable=True,
                     food_spawn_per_step=14)
    world = GridWorld(preset, seed=17, rng=np.random.default_rng(17))
    assert world.cfg is preset
    assert type(pickle.loads(pickle.dumps(preset))) is EnvConfig
    assert MLPPolicy is LegacyMLPPolicy
    torch.manual_seed(973)
    policy = MLPPolicy(obs_size=4, hidden_size=12, n_actions=3)
    cloned = MLPPolicy(obs_size=4, hidden_size=12, n_actions=3)
    clone_parameters(policy, cloned)
    obs = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
    assert torch.equal(policy(obs), cloned(obs))
    assert type(pickle.loads(pickle.dumps(policy))) is MLPPolicy
    assert RegistryAxis is LegacyRegistryAxis
    axis = RegistryAxis("variant", ["control", "changed"],
                        {"control": [], "changed": []})
    assert pickle.loads(pickle.dumps(axis)) == axis
    resources = {name: hashlib.sha256(read_resource_bytes(name)).hexdigest()
                 for name in ("crca.run-attempt-edge-cases.v1", "crca.g0-ratchet-reconciliation.v1")}
    validation = validate_science_request_v1(args.request)
    assert validation["ok"], validation
    print(json.dumps({
        "status": "INSTALLED_COMPONENTS_VERIFIED",
        "python": sys.version, "prefix": str(prefix), "module_origins": origins,
        "resources": resources, "request_validation": validation,
        "world_preset_construction": "PASS", "bounded_mlp_extension": "PASS",
        "legacy_identity_and_pickle": "PASS",
        "evolution_executed": False, "checkpoint_continuation_verified": False,
        "qualification_granted": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
