# Security and private data

This is research software. Treat experiment inputs, archives, extensions and
restorable states as trusted local inputs; do not expose operator services to
an untrusted network. The CPU walkthrough needs no credentials or network
service after dependencies have been installed.

Do not post secrets, personal information, private logs or unpublished research
records in a public issue. For a sensitive report, use GitHub's private security
reporting feature if enabled. If no private channel is available, open a minimal
issue requesting a contact route without including the sensitive details.

Public CI has read-only repository permissions and requires no repository
secrets. Publication scans cover the source manifest, commit identities and all
reachable Git blobs. They reduce disclosure risk; they are not a proof that
software or all future contributions are free of security defects.
