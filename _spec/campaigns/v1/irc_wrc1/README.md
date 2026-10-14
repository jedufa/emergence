# IRC-WRC-1 rehearsal inputs

The filenames retain the initial `g1` planning label, but every YAML now
declares the actual minimum admitted three-generation resident window.  The
campaign IDs and runtime roots use `g3`; consumers must use those semantic
identifiers, not infer a generation count from the filename.

These inputs are engineering rehearsals only.  Their separate `irc-wrc1-*`
runtime roots must never be read as final WRC qualification evidence.

`p057-reversed-g1.yaml` uses ordered pre-primary technical-gate selectors to
reverse the physical B2 canary dispatch order.  Matrix values themselves are
intentionally canonicalized and cannot express an order-dependent physical ABI
probe.  The rejected staged-primary attempt is retained under its separate
runtime root as a workflow limitation record, not as WRC evidence.
