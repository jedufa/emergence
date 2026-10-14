# WorldKit configuration ownership

WorldKit owns the environment configuration dataclasses used by GridWorld.
`EnvConfig` lives in `emergence_worldkit.config`; the focused injury, health,
energy, reproduction, biome, world, and spatial-ecology section dataclasses live
in `emergence_worldkit.config_sections`. Existing `engine.env.env_config*`
modules remain compatibility imports and preserve the `env.env_config*`
pickle addresses.

```python
from emergence_worldkit import EnvConfig

config = EnvConfig(width=32, height=32)
```

Configuration definitions use only the Python standard library. GridWorld
mechanics remain under `engine.env.gridworld_pkg` and use NumPy at runtime.
Configuration objects carry values only; they do not own simulation or random
state.

Status: engineering ownership refactor. The public defaults, field order, and
types are checked against baseline `fb00e803`; this change makes no new
scientific or qualification claim.

Focused validation:

```bash
python -m pytest -q tests/modular_contracts/test_worldkit_configuration_ownership.py
```

Bounds: this module owns configuration data and compatibility imports. It does
not move GridWorld mechanics, providers, RNG behavior, schemas, execution
contracts, or package metadata.
