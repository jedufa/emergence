# Contributing

Start with the component smoke and bounded CPU walkthrough in QUICKSTART.md.
State whether a change affects ecology, genetics, RNG addressing, checkpoint
bytes, execution identity, admission, storage or observation-only reporting.
Keep source code separate from runtime state and private research records.

Use matched controls and independent seed blocks for scientific comparisons.
Do not describe engineering smoke checks as scientific evidence or inherited
Qualification. Preserve upstream notices and disclose the source of copied code.

Run `python tools/release/validate_cpu.py` for the supported representative CPU
gate. The complete historical test tree includes specialized artifact and CUDA
requirements; see docs/release/VALIDATION.md before interpreting its results.
The historical architecture ratchet remains visible and is not silently reset.

The public source manifest binds every tracked file except the manifest itself.
Maintainers regenerate it after reviewing a change. Privacy exceptions bind an
exact file hash, line and rule; changed files require renewed review. Never add
an exception merely to make a scanner green. Report credentials and personal
data privately rather than posting them in an issue or attaching raw logs.
