"""tools.diagnostics

Verification and smoke-test helpers.

These scripts must not mutate simulation determinism-critical artifacts.
They are intended to be run by operators to detect regressions in the WebUI
and control-plane visibility.

"""

from __future__ import annotations
