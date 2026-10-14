# Harness Run Directory Structure

This document explains the meaning of each directory level in a full run path produced by the Emergence experiment harness.

Example path:

```
/runs/experiments/validation_2x2_devcoupling_latecoh/v1/9fa223aea5b5__72c7ac7/20260209_214713__b6db976__nox_smoke/runs/validation_2x2_devcoupling_latecoh__dc_False__lc_False__seed000042/metrics.csv
```

---

## 1. `/runs`

Top-level container for **all executed simulations**.

- This directory contains every run ever executed: smoke, quick, nightly, validation, aborted, or completed.
- No semantic meaning beyond *storage and containment*.

If it ran, it lives somewhere under `/runs`.

---

## 2. `/runs/experiments`

Indicates that the run was launched via a **declared experiment registry**.

- This distinguishes scientific experiment runs from ad-hoc debugging or manual invocations.
- Everything under this path corresponds to an explicit experimental intent.

Think: *"this run was planned"*.

---

## 3. `validation_2x2_devcoupling_latecoh`

The **experiment family name**, taken directly from the registry YAML.

- Represents one conceptual experiment or hypothesis.
- May be executed many times across days, machines, or code revisions.

Analogy: the title of a figure or table in a paper.

---

## 4. `v1`

The **experiment version**.

- Versioning applies to experimental intent and structure, not code.
- If the meaning of the experiment changes (axes, question, scope), this version must change.

This is not a Git version.

---

## 5. `9fa223aea5b5__72c7ac7`

The **registry materialization hash**.

This fingerprint uniquely identifies:
- The registry YAML contents
- Axis definitions
- Defaults such as generations, steps, population size

Two hashes are stored:
- First: raw registry content hash
- Second: normalized/resolved hash

If this directory differs, the experiment definition is **not identical**, even if the name is the same.

---

## 6. `20260209_214713__b6db976__nox_smoke`

The **execution envelope**.

Breakdown:
- `20260209_214713` — wall-clock launch timestamp
- `b6db976` — Git commit hash at execution time
- `nox_smoke` — invocation context (smoke / quick / nightly)

This answers:
- When was this run executed?
- With which code?
- Under what execution mode?

---

## 7. `/runs` (inner)

Separates **experiment orchestration artifacts** from **simulation outputs**.

- Outer directories describe *what was planned and launched*.
- Everything below this point is *simulation physics and results*.

This boundary matters for archiving and data transfer.

---

## 8. `validation_2x2_devcoupling_latecoh__dc_False__lc_False__seed000042`

A **single concrete evolutionary run**.

Decoded:
- Experiment name
- Axis assignments:
  - `dc_False` — developmental coupling disabled
  - `lc_False` — late-life coherence cost disabled
- `seed000042` — RNG seed

Properties:
- Fully self-contained
- Independently analyzable
- Safe to delete without affecting other runs

If this folder disappears, exactly one datapoint disappears.

---

## 9. `metrics.csv`

The **ground truth output**.

Contains:
- Per-generation measurements
- No inferred meaning
- No interpretation

All conclusions must trace back to rows in this file (or related outputs).

---

## Summary Mental Model

From top to bottom:

1. Storage universe
2. Declared scientific intent
3. Hypothesis family
4. Conceptual versioning
5. Exact experiment definition
6. Execution context
7. Simulation boundary
8. One world, one seed, one fate
9. Measured reality

The structure may look verbose, but nothing is redundant. It exists to prevent silent drift and accidental equivalence between non-identical experiments.

