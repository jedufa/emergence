"""A relocation cannot create an unbounded architecture exception."""
import hashlib
import json

import pytest

from tools.architecture.eiel_modular_import_guard_v1 import (
    GuardConfigurationError, audit_repository, check_repository,
)
from tests.modular_contracts.test_modular_import_guard_v1 import _baseline, _write


def _alias(root, *, body=None):
    body = body or 'from emergence_worldkit.config import Config\n__all__ = ["Config"]\n'
    _write(root, "engine/env/config.py", body)
    _write(root, "engine/simulation/run.py", "from env.config import Config\n")
    _write(root, "emergence_worldkit/config.py", "class Config: pass\n")
    row = {"path": "engine/env/config.py", "target": "emergence_worldkit.config",
           "names": ["Config"], "sha256": hashlib.sha256(body.encode()).hexdigest()}
    _write(root, "tools/architecture/eiel_modular_compatibility_aliases_v1.json",
           json.dumps({"schema_version": "eiel_modular_compatibility_aliases_v1", "aliases": [row]}))


def test_reexports_are_visible_without_increasing_legacy_ceilings(tmp_path):
    _alias(tmp_path)
    result = check_repository(tmp_path, _baseline({("simulation", "env"): 1}))
    assert result["status"] == "PASS"
    assert result["audit"]["resolved_internal_import_statements"] == 2
    assert result["audit"]["compatibility_edges"] == [
        {"source": "env", "target": "emergence_worldkit", "count": 1}]


@pytest.mark.parametrize("body", [
    'from emergence_worldkit.config import *\n__all__ = ["Config"]\n',
    'from emergence_worldkit.config import Config\n__all__ = ["Config"]\nprint("side effect")\n',
    'from emergence_worldkit.config import Config as Config\n__all__ = ["Config"]\n',
])
def test_even_rehashed_non_alias_code_fails_closed(tmp_path, body):
    _alias(tmp_path, body=body)
    with pytest.raises(GuardConfigurationError):
        audit_repository(tmp_path, audit_roots=["engine"])


def test_unregistered_and_changed_shims_fail(tmp_path):
    _alias(tmp_path)
    _write(tmp_path, "engine/env/config.py", 'from emergence_worldkit.config import Config\n')
    with pytest.raises(GuardConfigurationError, match="content changed"):
        audit_repository(tmp_path, audit_roots=["engine"])


def test_canonical_owner_is_still_subject_to_target_dag(tmp_path):
    _alias(tmp_path)
    _write(tmp_path, "emergence_worldkit/config.py", "import simulation.run\nclass Config: pass\n")
    result = check_repository(tmp_path, _baseline({("simulation", "env"): 1}))
    assert any(f["code"] == "FORBIDDEN_TARGET_DEPENDENCY" for f in result["findings"])


def test_missing_owner_and_unaudited_alias_fail_closed(tmp_path):
    _alias(tmp_path)
    (tmp_path / "emergence_worldkit/config.py").unlink()
    with pytest.raises(GuardConfigurationError, match="not audited"):
        audit_repository(tmp_path, audit_roots=["engine"])


def test_composition_cannot_hide_a_legacy_cycle_as_a_compatibility_alias(tmp_path):
    _alias(tmp_path)
    manifest = tmp_path / "tools/architecture/eiel_modular_compatibility_aliases_v1.json"
    raw = json.loads(manifest.read_text())
    raw["aliases"][0]["target"] = "eiel_composition.config"
    manifest.write_text(json.dumps(raw))
    with pytest.raises(GuardConfigurationError, match="legacy-dependent"):
        audit_repository(tmp_path, audit_roots=["engine"])
