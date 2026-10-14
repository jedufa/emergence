# Run the bounded CPU example

This walkthrough uses the real native Patch engine through `eiel execute`.
It checks one paired block and exact checkpoint continuations. It does not
claim scientific replication, general evolvability, or GPU Qualification.

## 1. Install

Use Linux x86-64, Python 3.14, a C++ compiler and Ninja. Start at the root of a
fresh checkout. Install CPU PyTorch first to avoid a CUDA dependency download.

```bash
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -r requirements-cpu.txt
python -m pip install ".[research,dev]"
python -m build --outdir dist
python -m pip install --force-reinstall --no-deps dist/emergence-0.1.0a1-py3-none-any.whl
python -I examples/release/component_smoke.py
```

Expected smoke status: `COMPONENT_SMOKE_VERIFIED`. This step alone does not run
an evolving population.

## 2. Choose fresh storage and build the native owner

Keep runtime files outside the source tree. Use a new directory for each demo;
the generator rejects existing request/run directories. The example reserves
at most 2 GiB for run artifacts, in addition to the storage policy's 4 GiB
filesystem reserve and measured uncertainty margin. Allow extra space for the
Python environment and build outputs.

```bash
export EIEL_RUNTIME_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/emergence-cpu.XXXXXX")"
export EIEL_ACTOR_STATE_ROOT="$EIEL_RUNTIME_ROOT/actors"
export EIEL_SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
export MAX_JOBS=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -I -m tools.scripts.build_eiel_resident_state_owner_extension \
  --build-dir "$EIEL_RUNTIME_ROOT/native/build" \
  --output "$EIEL_RUNTIME_ROOT/native/cpu-native-owner.so"
export EIEL_RESIDENT_STATE_OWNER_EXTENSION="$EIEL_RUNTIME_ROOT/native/cpu-native-owner.so"
```

The builder binds native source bytes, compiler and binary hashes. Do not reuse
an extension after changing its source. Set environment roots before starting
Python: several consumers intentionally cache their configuration at import.
The default runtime root is the current user's `.local/share/emergence`; explicit
`EIEL_RUNTIME_ROOT` and `EIEL_ARTIFACT_ROOT` override it. This example sets its own
actor root and does not require a shared GPU lease.

## 3. Materialize and execute the pair

```bash
python -I examples/release/paired_cpu_reproduction.py \
  --wheel dist/emergence-0.1.0a1-py3-none-any.whl \
  --request-dir "$EIEL_RUNTIME_ROOT/requests" \
  --run-root "$EIEL_RUNTIME_ROOT/paired-run" \
  --campaign-id public-cpu-demo
python -I -m tools.eiel execute "$EIEL_RUNTIME_ROOT/requests/request.json" --human
```

The generated request declares the CPU backend and legacy-generation engineering
exception explicitly. It does not silently fall back from an unavailable GPU.
The run includes two 960-tick treatments, two midpoint continuations, and a
reporting-parity check. A successful run produces a sealed result and lifecycle
inventory. Inspect the actual status; command invocation alone is not success.

## 4. Verify the recorded observations

Use the request, result and disposition digests captured from the completed
producer's receipts as trusted anchors. The read-only comparator accepts them:

```bash
python -I -m engine.campaigns.paired_cpu_usage_v1 \
  --run-root "$EIEL_RUNTIME_ROOT/paired-run" \
  --expected-request-sha256 "$TRUSTED_REQUEST_SHA256" \
  --expected-result-sha256 "$TRUSTED_RESULT_SHA256" \
  --expected-disposition-sha256 "$TRUSTED_DISPOSITION_SHA256"
```

These variables are values you retain independently from the producer receipts,
not arbitrary hashes generated to approve modified observations. The comparator
checks controls, seed matching, artifact seals, checkpoint continuations and
reporting parity. Missing or changed evidence fails closed.

## 5. Finish the run

Write a self-contained report with nonempty `Inputs`, `Settings`, `Results`,
`Limitations`, and `Reproduction` sections before reclaiming temporary simulation
output.
The canonical lifecycle command verifies the terminal run and exact root:

```bash
python -I -m tools.eiel storage finish-run "$EIEL_RUNTIME_ROOT/cpu-demo-report.md" \
  --run-root "$EIEL_RUNTIME_ROOT/paired-run" \
  --source-commit "$EIEL_SOURCE_GIT_COMMIT" --run-id public-cpu-demo --apply
```

Use the actual run identifier if you changed it. Review the lifecycle receipt.
Unfinished runs, unresolved consumers, symlinks and unclassified artifacts are
not permission for a broad filesystem cleanup.

See [VALIDATION.md](docs/release/VALIDATION.md) for the current verification record
and the exact scope of the supported CPU test gate.
