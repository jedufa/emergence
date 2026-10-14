# AgentKit policy module

`emergence_agentkit.policy` owns the current feed-forward `MLPPolicy` and the
in-place `clone_parameters` helper. The legacy `agents.agent_policy` import
continues to resolve to the same classes and historical pickle address.

```python
from emergence_agentkit import MLPPolicy

policy = MLPPolicy(obs_size=64, hidden_size=32, n_actions=5)
```

The module depends on PyTorch. Each policy instance owns its network parameters;
PyTorch's normal layer initialization uses the active PyTorch RNG. There is no
separate model registry or global policy state.

Status: canonical ownership of the existing MLP policy implementation.
Focused validation:

```bash
python -m pytest -q -o addopts='' tests/modular_contracts/test_agentkit_lab_ownership.py
```

Bounds: this surface defines the shown MLP and parameter-copy helper. It does
not define an arbitrary native custom-model loading or plugin contract. Model
execution authority remains with the canonical EIEL Launchpad route.
