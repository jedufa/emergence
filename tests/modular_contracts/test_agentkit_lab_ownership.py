"""Compatibility and RNG parity of extracted implementation owners."""
import importlib.util
import pickle
from pathlib import Path
import hashlib
import json
import sys

import torch

from emergence_agentkit import MLPPolicy, clone_parameters
from emergence_lab import RegistryAxis


def test_policy_legacy_aliases_and_full_pickle_identity():
    from agents.agent_policy import MLPPolicy as legacy
    from engine.agents.agent_policy import MLPPolicy as qualified
    assert MLPPolicy is legacy is qualified
    policy = MLPPolicy(4, 8, 3)
    assert type(pickle.loads(pickle.dumps(policy))) is MLPPolicy


def test_policy_matches_frozen_implementation_parameters_outputs_and_rng(tmp_path):
    root = Path(__file__).resolve().parents[2]
    fixture = root / "tests/fixtures/release_baseline"
    source = (fixture / "agent_policy_v1.py").read_bytes()
    provenance = json.loads((fixture / "agent_policy_provenance.json").read_text())
    assert hashlib.sha256(source).hexdigest() == provenance["sha256"]
    path = tmp_path / "frozen_policy.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("frozen_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(973)
    original = module.MLPPolicy(4, 8, 3)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(973)
    moved = MLPPolicy(4, 8, 3)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in original.state_dict().items():
        assert torch.equal(value, moved.state_dict()[name])
    observation = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
    assert torch.equal(original(observation), moved(observation))
    copy = MLPPolicy(4, 8, 3)
    clone_parameters(moved, copy)
    assert torch.equal(copy(observation), moved(observation))


def test_lab_registry_has_one_serialized_identity():
    from experiments.harness.schema import RegistryAxis as legacy
    assert legacy is RegistryAxis
    axis = RegistryAxis("variant", ["control", "changed"], {"control": [], "changed": []})
    restored = pickle.loads(pickle.dumps(axis))
    assert type(restored) is RegistryAxis
    assert restored == axis
