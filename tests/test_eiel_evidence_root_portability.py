"""Evidence-root configuration must never silently select a developer's machine."""
import json
from pathlib import Path

import pytest

from tests.evidence.cogp1_fixtures import cogp1_manifest
from tools.actors import evidence_manager as manager
from tools.actors.common import ActorError


def test_missing_configuration_fails_without_creating_files(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('EIEL_EVIDENCE_ROOT', raising=False)
    monkeypatch.chdir(tmp_path)
    assert manager.main(['discover']) == 2
    assert 'evidence_root_not_configured' in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('value', ['', '   ', 'relative/evidence'])
def test_invalid_environment_root(value, monkeypatch):
    monkeypatch.setenv('EIEL_EVIDENCE_ROOT', value)
    with pytest.raises(ActorError):
        manager._configured_evidence_root(None)


def test_environment_read_at_call_time_and_explicit_override(tmp_path, monkeypatch):
    first, second = tmp_path / 'first', tmp_path / 'second'
    monkeypatch.setenv('EIEL_EVIDENCE_ROOT', str(first))
    assert manager._configured_evidence_root(None) == first
    monkeypatch.setenv('EIEL_EVIDENCE_ROOT', str(second))
    assert manager._configured_evidence_root(None) == second
    monkeypatch.chdir(tmp_path)
    assert manager._configured_evidence_root('first') == first
    assert not first.exists() and not second.exists()


def test_file_is_not_a_root(tmp_path, monkeypatch):
    path = tmp_path / 'file'; path.write_text('test')
    monkeypatch.setenv('EIEL_EVIDENCE_ROOT', str(path))
    with pytest.raises(ActorError):
        manager._configured_evidence_root(None)


def test_discover_uses_environment(tmp_path, monkeypatch, capsys):
    manifest = cogp1_manifest(tmp_path / 'campaign', signal='supported')
    canonical = manifest.parent / 'campaign_evidence_manifest.json'
    canonical.write_bytes(manifest.read_bytes())
    monkeypatch.setenv('EIEL_EVIDENCE_ROOT', str(tmp_path))
    assert manager.main(['discover']) == 0
    assert json.loads(capsys.readouterr().out)['count'] >= 1


def test_assess_without_root_never_generates_report(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('EIEL_EVIDENCE_ROOT', raising=False)
    manifest = cogp1_manifest(tmp_path / 'campaign', signal='supported')
    monkeypatch.setattr(manager, 'generate_campaign_decision_report',
                        lambda *a, **kw: pytest.fail('report generation reached'))
    state = tmp_path / 'state'
    assert manager.main(['assess', str(manifest), '--state-root', str(state)]) == 2
    assert 'evidence_root_not_configured' in capsys.readouterr().out
    assert not state.exists()


def test_explicit_output_does_not_require_root(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('EIEL_EVIDENCE_ROOT', 'invalid-relative-root')
    manifest = cogp1_manifest(tmp_path / 'campaign', signal='supported')
    output = tmp_path / 'reports'
    assert manager.main(['assess', str(manifest), '--state-root', str(tmp_path / 'state'),
                         '--out-dir', str(output), '--deterministic']) == 0
    assert output.is_dir()
    assert json.loads(capsys.readouterr().out)['assessment']['scientific_qualification_effect'] == 'none'
