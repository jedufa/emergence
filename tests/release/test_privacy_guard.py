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


@pytest.mark.parametrize("allowed_identity", [True, False])
def test_detached_head_history_is_checked(tmp_path, allowed_identity):
    def git(*args, identity="123+fixture@users.noreply.github.com"):
        return subprocess.check_output([
            "git", "-C", str(tmp_path), "-c", "user.name=fixture",
            "-c", f"user.email={identity}", *args,
        ], stderr=subprocess.PIPE)

    git("init", "-q", "--initial-branch=main")
    policy = tmp_path / "tools/release/privacy-exceptions.json"
    policy.parent.mkdir(parents=True)
    policy.write_text(json.dumps({"findings": []}))
    (tmp_path / "SOURCE_MANIFEST.json").write_text(json.dumps({"files": [
        {"path": "tools/release/privacy-exceptions.json",
         "sha256": hashlib.sha256(policy.read_bytes()).hexdigest()},
    ]}))
    git("add", ".")
    git("commit", "-qm", "public base")
    git("checkout", "--detach", "-q")
    identity = "123+fixture@users.noreply.github.com" if allowed_identity else "fixture" + "@gmail.com"
    git("commit", "--allow-empty", "-qm", "detached contribution", identity=identity)
    head = git("rev-parse", "HEAD").decode().strip()
    assert head not in git("rev-list", "--branches", "--remotes", "--tags").decode().splitlines()

    report = inspect(tmp_path)
    assert report["commits"] == 2
    assert report["status"] == ("PASS" if allowed_identity else "FAIL")
    assert report["findings"] == ([] if allowed_identity else [
        {"rule": "commit_identity", "commit": head},
    ])

    # The workflow handles synthetic merge refs; the guard still rejects them.
    git("update-ref", "refs/pull/1/merge", "HEAD")
    assert "unexpected_ref_namespace" in {f["rule"] for f in inspect(tmp_path)["findings"]}
