"""The publication guard must detect new content and manifest drift."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools.release.privacy_guard import content_findings, inspect


@pytest.mark.parametrize("value, rule", [
    ("/" + "home/fictional-user/secret", "account_path"),
    ("192." + "168.1.99", "private_ipv4"),
    ("fictional" + "@gmail.com", "personal_email"),
    ("ghp_" + "A" * 36, "provider_token"),
    ("https://chatgpt.com/" + "c/fictional-conversation", "conversation_url"),
])
def test_sensitive_classes_are_detected_without_echoing_values(value, rule):
    findings = content_findings("fixture.txt", value.encode())
    assert rule in {f["rule"] for f in findings}
    assert value not in json.dumps(findings)


def test_unreviewed_binary_and_archive_are_rejected():
    assert content_findings("output.zip", b"archive")[0]["rule"] == "unreviewed_binary_or_large_file"
    assert content_findings("source.py", b"\0")[0]["rule"] == "unreviewed_binary_or_large_file"


def test_reviewed_tree_detects_content_membership_and_stale_exceptions(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    policy = tmp_path / "tools/release/privacy-exceptions.json"
    policy.parent.mkdir(parents=True)
    policy.write_text(json.dumps({"findings": []}))
    code = tmp_path / "example.py"
    code.write_text("print('example')\n")
    files = [{"path": str(p.relative_to(tmp_path)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in [code, policy]]
    (tmp_path / "SOURCE_MANIFEST.json").write_text(json.dumps({"files": files}))
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    assert inspect(tmp_path, check_history=False)["status"] == "PASS"
    code.write_text("print('changed')\n")
    report = inspect(tmp_path, check_history=False)
    assert report["status"] == "FAIL"
    assert "manifest_mismatch" in {f["rule"] for f in report["findings"]}
    extra = tmp_path / "extra.txt"
    extra.write_text("extra")
    subprocess.run(["git", "-C", str(tmp_path), "add", "extra.txt"], check=True)
    assert "manifest_membership_mismatch" in {f["rule"] for f in inspect(tmp_path, check_history=False)["findings"]}
