# `emergence_contracts` module guide

`emergence_contracts` is the low-dependency public contract kernel for the
modular EIEL platform. Its top-level package re-exports the versioned Python
contract types; the package does not own runtime execution, scientific state,
checkpoint payloads, or qualification decisions.

## Public API and packaged resources

Import the public V1 contract types from `emergence_contracts`, for example
`SemanticRunSpecV1`, `RunOccurrenceV1`, `AttemptAdmissionRequestV1`, and
`AttemptStartV1`. These names are re-exported by the package root. Internal
modules describe implementation layout and are not needed by API consumers.

The `emergence_contracts.resources` subpackage provides the stable resource
API `read_resource_bytes(name)`. Its two allowlisted names are
`crca.run-attempt-edge-cases.v1` and
`crca.g0-ratchet-reconciliation.v1`. The accessor uses
`importlib.resources`, reads package data without assuming a checkout path, and
checks each payload against its pinned SHA-256. An unknown name raises
`KeyError`; a packaged file whose bytes do not match the pinned digest raises
`ResourceIntegrityError`. The adjacent `provenance.json` records each source
path, source digest, source commit `fb00e803`, and authority commit
`86477bdcd0702ddae3bfd27597bf2007dbd127a0`.

The resources are byte copies of the authoritative CRCA fixtures. The public
accessor is intentionally limited to these named fixtures; it is not a general
filesystem or package-path reader. Distribution metadata must include the
files under `emergence_contracts/resources/` in wheels and source archives.

## Phase 2 contract map

The following 18 contracts are indexed from historical internal document
`EIEL-DOC-002454`. That detailed mapping is not distributed; this table and the
packaged contract APIs define the contributor entry points in this snapshot:

| ID | Contract |
| --- | --- |
| C-001 | Canonical Identity Contract v1 |
| C-010 | Component Descriptor Contract v1 |
| C-020 | Runtime Services Contract v1 |
| C-030 | Execution Schedule and Run Cursor Contract v1 |
| C-040 | World Component Contract v1 |
| C-050 | Agent Component Contract v1 |
| C-060 | Adaptation Component Contract v1 |
| C-070 | System Assembly Contract v1 |
| C-080 | Snapshot Participant and Composite Checkpoint Contract v1 |
| C-090 | RNG Authority and Lease Contract v1 |
| C-100 | Runtime Event and Telemetry Contract v1 |
| C-110 | Runtime Engine and Execution Provider Contract v1 |
| C-120 | Experiment Plan and Executor Contract v1 |
| C-130 | Run Artifact Manifest Contract v1 |
| C-140 | Evidence Evaluator Contract v1 |
| C-150 | Capability and Qualification Contract v1 |
| C-160 | Control Command and Control Port Contract v1 |
| C-170 | Native Binding Contract v1 |

The mapping document remains the detailed ownership and Phase 2 disposition
source. This table is an index; it does not promote a contract's recommended
status or change the current state owner.

## Dependencies and imports

Runtime contract models and the resource accessor use only the Python standard
library. Resource loading specifically depends on `importlib.resources` and
`hashlib`; it does not read `_artifacts/` or depend on the current working
directory. Keep these APIs usable from an installed package as well as a source
checkout by shipping the resource files as package data.

Use package-root imports for contract types and the resource subpackage for
fixture bytes:

```python
from emergence_contracts import RunOccurrenceV1, SemanticRunSpecV1
from emergence_contracts.resources import read_resource_bytes

matrix = read_resource_bytes("crca.run-attempt-edge-cases.v1")
```

## State ownership and version amendments

The contracts describe identities, values, references, and boundaries. They do
not create a parallel state owner. Existing Runtime, WorldKit, AgentKit,
Adaptation, Lab, Evidence, and native owners remain authoritative as specified
by the mapping document. A DTO, descriptor, or adapter view must not duplicate
checkpoint state, RNG authority, execution lifecycle, or qualification
authority.

Keep V1 semantics and serialized field meanings stable. Make compatible,
optional additions only when old consumers retain their current behavior. A
semantic or wire incompatibility requires a separately named V2 contract and
an explicit adapter or migration plan; do not silently repurpose a V1 field.
Changes to the packaged CRCA fixtures require an authority-backed source
amendment, a new source digest and provenance entry, matching accessor pin,
and updated tests. Do not edit the canonical architecture ratchet baseline to
make a resource or contract change pass.

## Support and focused testing

The contract package is supported through its exported types, documented
resource names, versioned schemas, and focused modular-contract tests. Changes
should preserve byte-level resource verification and the existing semantic
assertions. Run the focused CRCA contract tests with:

```bash
python -m pytest -q -o addopts='' \
  tests/modular_contracts/test_crca_execution_identity_v1.py
```

For changes that affect module boundaries or the canonical ratchet, also use
the architecture checks required by the repository's CRCA procedure. The
baseline at `tools/architecture/baselines/eiel_modular_legacy_imports_v1.json`
is canonical and must not be weakened to accommodate implementation changes.
