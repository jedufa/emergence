"""Phase 3 membership harness entrypoint.

The Phase 3 E2E determinism script expects:

    python -m tests.harnesses.phase3_resume.harness.phase3_membership_harness ...

The implementation lives under tests.harnesses.phase3_resume.verify.phase3_membership_harness.
This shim keeps the import path stable.
"""

from tests.harnesses.phase3_resume.verify.phase3_membership_harness import main


if __name__ == "__main__":
    raise SystemExit(main())
