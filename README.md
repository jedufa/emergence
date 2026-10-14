# Emergence

**What if intelligence were not designed, but discovered by evolution?**

A computational laboratory for life, adaptation, and the conditions that make
adaptive capabilities useful.

Create the conditions. Let evolution explore. Follow the evidence.

Emergence brings together ecological worlds, heritable agent models, controlled
experiments, and authenticated execution records. Its scientific question is
which environmental and population conditions allow useful adaptations to arise,
persist, and spread. Intelligence, cooperation, and cultural evolution are
research ambitions—not results established by this distribution.

## Explore the laboratory

| Area | What the source provides | Start here |
|---|---|---|
| Worlds | Resource landscapes, environmental configuration, ecological dynamics | [WorldKit](emergence_worldkit/), [world implementations](engine/env/) |
| Agents and adaptation | Policies, heredity, mutation, selection and experimental adaptation mechanisms | [AgentKit](emergence_agentkit/), [evolution](engine/evolution/) |
| Experiments | Matched treatments, deterministic campaign compilation, explicit execution requests | [Lab](emergence_lab/), [composition](eiel_composition/) |
| Evidence | Typed identities, checkpoint contracts, artifact inventories and paired comparisons | [Contracts](emergence_contracts/), [evidence](emergence_evidence/), [campaigns](engine/campaigns/) |

```mermaid
flowchart LR
    Q[Research question] --> C[Explicit campaign and controls]
    C --> W[Worlds and evolving populations]
    W --> O[Authenticated observations]
    O --> A[Comparison and exact continuation]
    A --> Q
```

The source includes both current modular interfaces and legacy integrated runtime
components. The boundaries are still being improved; this is an early research
source preview, not a stable general-purpose simulation API.

## Try it

The verified beginner route is Linux x86-64 with Python 3.14 and CPU-only PyTorch.
A C++ compiler and Ninja are needed for the native example. No GPU, cloud account,
credentials, or historical research archive is required.

```bash
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -r requirements-cpu.txt
python -m pip install ".[research,dev]"
python -I examples/release/component_smoke.py
```

The smoke constructs real world configurations and verifies packaged contract
resources. It does not simulate evolution. Continue with the
[CPU experiment walkthrough](QUICKSTART.md) to build the native engine, execute
one matched treatment pair, and check exact midpoint continuations.

That example changes reproduction-attempt probability from 0 to 0.15 while
holding the declared seeds and other inputs fixed. It runs six legacy
generations of 160 ticks and restores each treatment from its midpoint. It is a
bounded engineering demonstration using the explicit legacy route, not evidence
of general evolvability or a population-structure experiment. Two treatments
sharing seeds are one paired block, not two independent replications.

## Use the components

```python
from dataclasses import replace
from emergence_worldkit import EnvConfig
from emergence_agentkit import MLPPolicy

world = replace(EnvConfig(), width=32, height=32, patch_world_enable=True)
policy = MLPPolicy(obs_size=4, hidden_size=12, n_actions=3)
```

See [validation and supported scope](docs/release/VALIDATION.md) for what was
actually checked, [source changes](docs/release/PUBLIC_SCOPE.md) for the public
snapshot boundary, and [third-party notices](THIRD_PARTY_NOTICES.md) for source
attribution and dependency terms.

## Research with care

Survival, retained adaptive variation, and the ability to generate new
adaptations are different outcomes. A population that recovers after a shock
has not thereby demonstrated intelligence or greater evolvability. Experiments
should declare controls, independent seed blocks, intervention schedules,
measurement rules, and storage budgets before execution.

Exact history is useful only when checkpoint identity, implementation identity,
and restoration checks support the claim. This source distribution grants no
inherited GPU, Full Evolution, or formal Qualification status. Advanced CUDA,
cluster, historical operator, and long-running campaign workflows need their
own compatible artifacts, configuration and validation. Some historical test
sentinels remain outside the supported CPU gate; their limitations are documented.

## Participate

Explore a world or agent model, reproduce the CPU example, question an
interpretation, or propose a bounded experiment. Changes should preserve
reproducible identities and state clearly which assumptions they introduce.
See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## License

Eligible first-party code, tests and documentation are offered under
[Apache-2.0](LICENSE), Copyright (c) 2026 Jean-Francois Dufour.
Included upstream material retains its own terms, including BSD, the Sun fdlibm
notice and LGPL-2.1-or-later. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The source license does not grant rights to third-party trademarks, datasets,
CUDA toolchains or binaries that are not included here.
