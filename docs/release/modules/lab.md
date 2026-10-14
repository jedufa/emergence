# Lab registry module

`emergence_lab.registry` owns the typed experiment registry records and
`load_and_validate_registry`, which validates and normalizes a mapping before
the harness consumes it. The legacy `experiments.harness.schema` module remains
a compatibility import for these classes and helpers.

```python
from emergence_lab import load_and_validate_registry

registry = load_and_validate_registry({
    "harness_api": 1,
    "version": 1,
    "name": "example",
    "matrix_order": ["mode", "seed"],
    "axes": {
        "mode": {
            "values": ["control", "variant"],
            "flags": {"control": [], "variant": ["--mode", "variant"]},
        },
    },
    "seeds": [1],
    "cli_defaults": [[]],
    "expected_artifacts": ["manifest.json"],
    "naming": {
        "run_tag_prefix": "example",
        "run_id_format": "{name}-{seed}",
        "run_tag_format": "example-{seed}",
    },
})
```

The registry module uses only the Python standard library. It returns frozen
dataclass records, with nested list and dictionary values retained as ordinary
mutable data; it has no process-global registry state. YAML parsing and harness
execution belong to their caller.

Status: canonical registry validation and normalization module.
Focused validation:

```bash
python -m pytest -q -o addopts='' tests/modular_contracts/test_agentkit_lab_ownership.py
```

Bounds: this module validates registry declarations; it does not launch work.
Execution stays on the canonical EIEL Launchpad route. It does not define an
arbitrary native custom-model interface.
