<div align="center">

# EMERGENCE

### A research platform for evolution, adaptive systems, and reproducible experiments

**Build worlds. Evolve populations. Test hypotheses. Trace the evidence.**

Emergence is an open-source computational laboratory for exploring how useful behaviours arise from the interaction of **agents, heredity, environments, and selection**. It is also the reusable engineering platform built to investigate those questions: simulation components, experiment definitions, execution contracts, checkpoints, replay, and evidence analysis.

[**Explore the platform ↓**](#a-platform-not-just-a-simulation) · [**See what you can reuse →**](#the-reusable-building-blocks) · [**Run the CPU example →**](#get-started) · [**Understand the limits →**](#what-is-verified)

<br>

![Status](https://img.shields.io/badge/status-research%20source%20preview-425c69?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.14%20CPU%20route-3776ab?style=flat-square)
![License](https://img.shields.io/badge/license-Apache--2.0-2e8474?style=flat-square)
![Evidence](https://img.shields.io/badge/claims-evidence--bounded-6e6786?style=flat-square)

</div>

![Conceptual visual identity for the Emergence laboratory: ecological variation connected to experimental evidence](docs/figures/emergence-lab.svg)

<sub>Conceptual illustration, not simulation output. Actual release verification is documented in [VALIDATION.md](docs/release/VALIDATION.md).</sub>

---

## The idea: build the world, not the intelligence

Most AI projects start by designing a system to perform a task. Emergence asks a different question:

> **What kinds of environments and evolutionary pressures make adaptive capabilities worth developing?**

Agents encounter resources, danger, spatial constraints, and reproductive trade-offs. Their behaviours and inherited properties can vary. Rather than directly rewarding an "intelligence score", experiments change the conditions under which memory, restraint, exploration, or other capabilities might become useful.

The scientific ambition is to investigate progressively richer adaptation, potentially including learning, cooperation, and cultural transmission. **Those are research questions, not achievements claimed by this repository.** A good-looking trajectory is not a substitute for causal evidence.

## A platform, not just a simulation

The scientific project motivated an engineering problem: how do you run complicated, evolving-world experiments **without losing control of what was run, what changed, and what the results actually mean?**

Emergence addresses that problem with multiple layers that can be studied, adapted, or reused independently of its original research agenda:

- **Configurable artificial worlds:** ecological landscapes, resource dynamics, environmental pressures, and alternative world conditions.
- **Evolving populations:** agent policies, genomes, mutation, reproduction, selection, and lineage-related mechanics.
- **Declarative experiments:** research questions expressed through settings, parameter axes, seed assignments, controls, and execution requests rather than a separate program for every hypothesis.
- **Multi-world campaign machinery:** expand parameter matrices into logically identified worlds and physical batches, with native CPU and advanced CUDA-related source paths. The public verification boundary is narrower than those advanced routes.
- **A composed execution stack:** one production lifecycle in the architecture, with component configuration and native execution machinery instead of a new simulator for each experiment.
- **Reproducible histories:** deterministic contracts, source and configuration identities, checkpoint/restore, continuation, and replay-oriented verification.
- **Evidence as a first-class output:** paired comparisons, recorded observations, provenance checks, and analysis that remains separate from the mechanisms being tested.
- **Controlled research operations:** resource and artifact lifecycle contracts, validation gates, and explicit decisions about what the evidence does or does not establish.

```mermaid
flowchart LR
    Q["Question and controls"] --> D["Declarative experiment"]
    D --> E["Execution and admission"]
    E --> W["Worlds, agents, evolution"]
    W --> P["Checkpoints and observations"]
    P --> A["Comparison and analysis"]
    A --> R["Bounded result"]
    R --> Q
```

*Conceptual architecture. The released code contains both modular interfaces and integrated legacy components. This is not a promise of a stable, general-purpose plug-and-play API.*

## The reusable building blocks

The repository exposes real source boundaries, not just diagrams. Different parts have different levels of packaging and runtime maturity.

| Building block | What it offers | Start here |
|:--|:--|:--|
| **WorldKit** | World and ecology configuration, including the Patch-world feature surface | [`emergence_worldkit/`](emergence_worldkit/) · [`engine/env/`](engine/env/) |
| **AgentKit** | A reusable PyTorch MLP policy and parameter-cloning interface, alongside the engine's wider agent and evolutionary mechanics | [`emergence_agentkit/`](emergence_agentkit/) · [`engine/agents/`](engine/agents/) · [`engine/evolution/`](engine/evolution/) |
| **Lab** | Experiment registries, typed axes, seed validation, and deterministic declaration patterns | [`emergence_lab/`](emergence_lab/) · [`engine/campaigns/`](engine/campaigns/) |
| **Contracts** | Typed scientific and runtime interfaces, canonical identities, execution requests, and checkpoint/evidence schemas | [`emergence_contracts/`](emergence_contracts/) |
| **Composition** | Integration adapters and scientific request-to-execution machinery | [`eiel_composition/`](eiel_composition/) |
| **Evidence** | Reducer interfaces, campaign analysis, evidence publication, and source-bound result structures | [`emergence_evidence/`](emergence_evidence/) |
| **Simulation engine** | Ecological and evolutionary runtime code, native execution sources, and optional GPU-related implementation paths | [`engine/`](engine/) |

These are **reusable foundations**, not separately versioned, compatibility-guaranteed products. WorldKit currently owns configuration rather than the world simulation loop; AgentKit's public surface is intentionally small; advanced composition and native routes require their own integration and validation.

### What could you use this for?

**Build a new artificial-life experiment.** Define alternative ecological pressures or behavioural settings, hold controls constant, and compare what actually changes.

**Investigate a mechanism causally.** Run matched treatments, disable one mechanism, or restore a known state to distinguish a plausible explanation from a coincidental outcome.

**Build better research infrastructure.** Adapt ideas and code for typed experiment requests, reproducible execution identities, checkpoint continuations, evidence processing, and careful artifact retention.

Not every example is a ready-made command. Some require extending the integrated engine, and anything beyond the verified CPU route must be tested in its own environment.

## Scientific method is part of the architecture

Several choices are deliberate, and useful well beyond evolutionary simulation:

1. **Change conditions, not success labels.** Richer environments can make capabilities useful without awarding direct bonuses for looking intelligent.
2. **Separate configuration from execution.** A new world, policy, or experiment should not automatically demand another production engine.
3. **Compare alternatives, not anecdotes.** Controls, paired seeds, ablations, and independent replication are designed to challenge explanations.
4. **Make history inspectable.** Execution identities, snapshots, continuation, lineage-related observations, and provenance help connect results to the conditions that produced them.
5. **Treat claims as bounded.** A technical validation proves that a particular route works under specified conditions; it does not prove an evolutionary theory.
6. **Preserve evidence without keeping everything forever.** Artifact identities and retention rules distinguish durable results from temporary execution material.

The goal is not merely to generate interesting worlds. It is to make those worlds **scientifically interrogable**.

## Get started

### 1. Explore the components

The verified beginner path is **Linux x86-64, Python 3.14, CPU-only PyTorch**. The smoke test checks importable component/configuration behaviour and packaged contract resources. It does **not** execute evolution.

```bash
git clone https://github.com/jedufa/emergence.git
cd emergence
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -r requirements-cpu.txt
python -m pip install ".[research,dev]"
python -I examples/release/component_smoke.py
```

### 2. Configure a world and agent policy

```python
from dataclasses import replace
from emergence_worldkit import EnvConfig
from emergence_agentkit import MLPPolicy

world = replace(EnvConfig(), width=32, height=32, patch_world_enable=True)
policy = MLPPolicy(obs_size=4, hidden_size=12, n_actions=3)
```

This constructs configuration and a policy, **not** a complete running evolutionary experiment.

### 3. Run a real paired CPU experiment

Follow the [**CPU quickstart**](QUICKSTART.md) to build the native CPU owner, execute two controlled treatments, verify midpoint checkpoint continuation, and inspect the evidence and storage-lifecycle receipts. You will need a C++ compiler and Ninja, but no GPU, API key, or historical evidence archive.

The verified example uses a **32 × 32 Patch world**, **256 initial agents**, and **512 capacity**, with reproduction-attempt probabilities **0** and **0.15** under one declared paired seed block. Each treatment runs six legacy generations of 160 ticks. Both midpoint continuations matched the recorded terminal evidence.

This is a meaningful **engineering demonstration**, not an independently replicated scientific result. Details and exclusions are in [the validation record](docs/release/VALIDATION.md).

## What is verified?

| Scope | Status in this public source snapshot |
|:--|:--|
| Component imports, configuration, packaged resources | **Locally checked** |
| Representative CPU tests and native paired example | **Locally checked**, with documented exceptions and exclusions |
| Midpoint checkpoint/continuation equality in the CPU example | **Verified for the stated source and environment** |
| Native/CUDA and larger-scale science implementation source | **Included in part**, not granted public GPU or full-evolution qualification |
| Historical campaigns, private scientific evidence, formal route qualification | **Not transferred to this public release** |
| General intelligence, broad evolvability, cooperation, cultural evolution | **Not established** |

The public snapshot intentionally excludes private operational histories, large campaign archives, machine-specific launch infrastructure, and historical qualification authority. Source availability does not make those absent assets reproducible. The historical full-tree architecture check also has recorded failures; the supported CPU gate is narrower. This matters more than an impressive badge.

Read [**Verified scope and limitations**](docs/release/VALIDATION.md) and [**Public source boundary**](docs/release/PUBLIC_SCOPE.md) before treating an advanced route as ready to run or citing a result.

## Where the research can go

The longer-term programme explores three connected questions: **how ecology creates demand for adaptation; how heredity and acquired capabilities survive across generations; and how population structure, social learning, or environmental inheritance might support cumulative change.**

These directions are motivations and hypotheses. Emergence does not claim to have evolved general intelligence, consciousness, culture, or open-ended complexity. The point of the platform is to make such claims difficult to assert without the necessary experiments.

## Contribute and reuse

Emergence welcomes reproducible examples, new ecological questions, focused fixes, and experiments with clear controls and limitations. Start with the [contribution guide](CONTRIBUTING.md) and the [CPU quickstart](QUICKSTART.md). For responsible reporting of security issues, see [SECURITY.md](SECURITY.md).

Eligible first-party source, tests, and documentation are available under **[Apache-2.0](LICENSE)**, Copyright © 2026 Jean-Francois Dufour. Included third-party material retains its own terms, including BSD, Sun fdlibm, and LGPL-2.1-or-later; see [third-party notices](THIRD_PARTY_NOTICES.md). The repository does not license absent data, CUDA toolchains, or third-party trademarks.

---

<div align="center">

### Build the world. Run the experiment. Respect the evidence.

*EMERGENCE · Evolutionary Intelligence Emergence Laboratory*

</div>
