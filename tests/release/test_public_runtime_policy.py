"""Portable roots must retain the storage admission and integrity boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns import storage_availability as storage
from engine.runtime_roots import public_runtime_path, RuntimeRootConfigurationError


def test_runtime_adapter_observes_explicit_environment_without_writes(tmp_path, monkeypatch):
    root = tmp_path / "fresh"
    monkeypatch.setenv("EIEL_RUNTIME_ROOT", str(root))
    assert public_runtime_path("build/cache") == str(root / "build/cache")
    assert public_runtime_path("artifacts/") == str(root / "artifacts") + "/"
    assert not root.exists()


@pytest.mark.parametrize("relative", ["/outside", "../outside", "inside/../../outside"])
def test_runtime_adapter_rejects_escape(relative):
    with pytest.raises(RuntimeRootConfigurationError):
        public_runtime_path(relative)


def test_template_binds_configured_root_and_keeps_budget(tmp_path):
    root = tmp_path / "runtime"
    code = '''
import json
from engine.campaigns.storage_availability import load_data2_storage_availability_policy_v1
print(json.dumps(load_data2_storage_availability_policy_v1()))
'''
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                            text=True, env={**os.environ, "EIEL_RUNTIME_ROOT": str(root)})
    policy = json.loads(result.stdout)
    assert policy["filesystem_root"] == str(root)
    assert policy["permanent_evidence_capsules"]["root"] == str(root / ".permanent-evidence")
    assert policy["reserve_model"]["filesystem_reserve_bytes"] == 4 * 1024**3
    assert policy["cold_diagnostic_profile_assets"]["admission_policy"] == "explicit_manifest_selected_only"
    digest = policy.pop("policy_sha256")
    assert digest == canonical_sha256(policy)
    assert not root.exists()


@pytest.mark.parametrize("rehash", [False, True])
def test_changed_budget_never_gains_authority(tmp_path, rehash):
    policy = json.loads(storage.CANONICAL_DATA2_STORAGE_AVAILABILITY_POLICY_PATH.read_text())
    policy["reserve_model"]["filesystem_reserve_bytes"] = 0
    if rehash:
        policy.pop("policy_sha256")
        policy["policy_sha256"] = canonical_sha256(policy)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(storage.StorageAvailabilityPolicyError):
        storage.load_data2_storage_availability_policy_v1(path)


def test_management_and_aggregate_roots_cannot_be_run_roots():
    for root in [storage.DATA2_RUNTIME_ROOT, storage.DATA2_PERMANENT_EVIDENCE_ROOT,
                 storage.DATA2_LIFECYCLE_MANAGEMENT_ROOT,
                 storage.DATA2_LIFECYCLE_MANAGEMENT_ROOT / "child"]:
        with pytest.raises(storage.StorageAvailabilityPolicyError):
            storage.validate_data2_run_root_v1(root)
    assert storage.validate_data2_run_root_v1(storage.DATA2_RUNTIME_ROOT / "runs" / "one")
