# Ops

Existing `tools.eiel`, actors, Resource Manager and exact artifact lifecycle remain the operator interfaces.

Example: Use `eiel resourceManager inspect --no-store --human` for read-only resource observation and `eiel science run` for fresh science.

Ops does not own biological state. Lifecycle contracts admit storage and release only exact authorized outputs. Private local service/remote integrations are retained source, not configured public services.

Dependencies follow [the source map](../source-to-module.json) and the existing ten-module DAG. Shared legacy imports remain under the unchanged debt ratchet.

Important checks: Existing admission/lifecycle tests and exact run receipts; no remote operation is part of release checks.

Status: Private deployment integration retained. No email, Bridge, upload, destructive cleanup or public publication is authorized.

Evidence Manager has no implicit machine-specific evidence directory. Pass
`discover --root PATH` or `assess --evidence-root PATH`, or set the absolute
`EIEL_EVIDENCE_ROOT` environment variable. Command-line roots take precedence;
relative command-line paths remain relative to the current working directory.
`assess --out-dir PATH` selects the report destination directly and does not
require an evidence root. Missing configuration fails before report generation.
Configuration does not authorize mutation of retained scientific evidence.
