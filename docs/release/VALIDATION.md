# Verified scope and limitations

This source preview is a bounded engineering release. It grants no scientific,
GPU, Full Evolution or formal Qualification claim.

## Local verification

Toolchain: Linux x86-64, Python 3.14.4, PyTorch 2.10.0+cpu, NumPy 2.3.5,
GCC 15.2.0, Ninja 1.13.2. Exact Python dependencies are in requirements-cpu.txt.
The native builder uses `-O2 -ffp-contract=off` and records source/compiler/binary
identities. Dependencies were installed into a fresh virtual environment.

| Check | Result |
|---|---|
| Component smoke | Real world configuration and packaged resource hashes verified |
| Installed components | Isolated wheel imports, module ownership, policy cloning, configuration/pickle identity and static request validation |
| Representative CPU gate | 396 passed, 1 skipped, 3 expected failures, 2 explicitly excluded historical sentinels |
| Native CPU pair | Canonical execution completed with mechanical PASS and declared engineering deviations |
| Exact continuations | Both midpoint restores matched metric-row hashes and terminal checkpoint digests |
| Reporting parity | Matched |
| Canonical run retirement | Self-contained report retained; exact transient run root reclaimed |
| Source/wheel | Built from the curated source; source manifest and package contents inspected |
| Privacy | Pattern/content/metadata checks plus independent Gitleaks scan; specific reviewed false positives are content-bound |

The CPU example uses one paired block with ecology seed 101 and task seed 1001,
a 32-by-32 Patch world, 256 founders and capacity 512. The treatments differ only
in reproduction-attempt probability (0 versus 0.15), over six legacy generations
of 160 ticks. Recorded births were 0 versus 256; survivors were 256 versus 510;
deaths were 0 versus 2. These are observations for this bounded example, not a
replicated evolutionary conclusion. Exact continuation concerns the tested
implementation and environment, not arbitrary toolchains.

The optional external Canopy relay was unavailable. Execution recorded degraded
observability explicitly; the authenticated producer observations, mechanical
checks and continuations remained available. No external service was started.

## What the CPU gate deliberately does not claim

Run `python tools/release/validate_cpu.py`. Its selected modules cover contracts,
packaging, public runtime/storage policy, evidence handling, security, exact-root
retirement and campaign analysis. Every exclusion is printed before pytest runs.

- One evidence-manager test requires historical raw campaign outputs that are
  not distributed. Synthetic evidence-manager checks remain active.
- Three existing expected failures require an unavailable native ACS artifact;
  they retain their exact exception check and do not count as executed semantic
  assertions.
- The frozen BundleV2 byte-hash sentinel fails identically on the untouched source
  baseline in this CPU environment. It remains unchanged in the tree, outside
  this gate. Actual continuation equality is independently checked in the native
  demo and other checkpoint tests.
- The historical full-tree architecture ratchet remains unchanged and **fails**.
  Curation removes old import edges and portability adds explicit runtime-root
  dependencies. Its latest report has 53 findings: 26 edge-growth findings and
  27 ceilings that have not been lowered. It reports zero target-package
  violations, zero forbidden contract imports and no parse errors. This is not
  an architecture-cleanliness pass; no historical ceiling was raised to hide it.

An additional historical phase-C/lifecycle/analysis run produced 353 passes and
9 failures. All nine failures were reproduced on the untouched source baseline
in the same environment. They concern legacy operator-budget assertions,
checkpoint-resume cases, and a resource-handling consumption assertion. Those
experimental suites remain visible and are outside the supported CPU gate;
their failures are not relabeled as successes. Full repository-wide pytest,
advanced CUDA workflows, GPU hardware parity, long campaigns and historical
qualification inheritance are not release claims.

A public CI run, once the repository exists, is separate evidence from these
local checks. CI uses read-only repository permissions, no repository secrets,
and no upload of raw experiment logs or artifacts.
