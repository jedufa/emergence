#!/usr/bin/env python3
"""tools.run_job

Compatibility entrypoint for the wrapper executor.

Why this file exists
--------------------
Parts of the tooling stack (including manifest runner) invoke the wrapper via:

    python -m tools.run_job ...

During a refactor, the shim ended up under tools/manifest/run_job.py, which means
`tools.run_job` no longer exists as an importable module, so the wrapper never
starts, no wrapper folder is created, and the manifest runner reports
RUN_EVIDENCE_WRITE_FAIL for every job.

This shim restores the expected module path without changing any behavior.

Implementation
--------------
Delegates to: tools.core.run_job.main
"""

from __future__ import annotations

from tools.core.run_job import main


if __name__ == "__main__":
    raise SystemExit(main())
