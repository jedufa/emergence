from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from engine.campaigns import root_retirement
from eiel_composition import science_run_report as reporting


@pytest.fixture
def setup(tmp_path, monkeypatch):
    runtime = tmp_path / 'causyl-runtime'
    runtime.mkdir()
    monkeypatch.setattr(root_retirement, 'CANONICAL_RUNTIME_ROOTS', frozenset({runtime}))
    monkeypatch.setattr(root_retirement, '_assert_live_state_clear', lambda root: None)
    monkeypatch.setattr(reporting, '_source_context', lambda repo: {'commit': 'a' * 40, 'source_patch': '', 'additional_source_files': {}})
    request = tmp_path / 'request.json'
    request.write_text(json.dumps({'campaign_id': 'repeatable-test', 'question': 'Test', 'run': {'total_ticks': 10}}))
    return request, dict(repo_root=tmp_path, owner_parent=runtime / 'owned', report_parent=tmp_path / 'reports', pending_root=runtime / 'pending')


def executor(request, **options):
    run = options['runtime_parent'] / 'run'
    mat = options['materialization_parent'] / 'plan'
    run.mkdir(parents=True)
    mat.mkdir(parents=True)
    aggregate = run / 'aggregate.json'
    aggregate.write_text(json.dumps({'decision': 'VALID_NEGATIVE', 'worlds': [{'raw': 'not retained'}], 'summary': {'surviving': 0}}))
    (run / 'checkpoint.bin').write_bytes(b'checkpoint')
    resolved = {}
    for key in ('base_campaign_plan', 'science_context'):
        path = mat / f'{key}.json'
        path.write_text(json.dumps({'identity': key, 'settings': {'ticks': 10}}))
        resolved[key] = {'path': str(path), 'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (mat / 'science_request_plan_receipt.json').write_text(json.dumps({'resolved_inputs': resolved}))
    return {'ok': True, 'status': 'COMPLETED', 'runtime_root': str(run), 'materialization_root': str(mat),
            'lifecycle_closeout': {'inventory_complete': True, 'terminal_status': 'completed'},
            'aggregate': {'result': {'path': str(aggregate), 'file_sha256': hashlib.sha256(aggregate.read_bytes()).hexdigest()}}}


def test_full_owner_cleanup_keeps_only_report_and_pending_pointer(setup):
    request, options = setup
    result = reporting.run_science_with_report_v1(request, executor=executor, **options)
    assert result['status'] == 'COMPLETED'
    assert list(options['owner_parent'].iterdir()) == []
    report = Path(result['report_path']).read_text()
    assert 'VALID_NEGATIVE' in report
    assert '"raw"' not in report
    assert 'total_ticks' in report
    assert 'Cleanup' in report
    assert len(list(options['pending_root'].glob('*.json'))) == 1
    assert [p.suffix for p in options['report_parent'].iterdir()] == ['.md']


def test_unfinished_run_not_deleted_or_reported_completed(setup):
    request, options = setup
    def waiting(request, **kwargs):
        value = executor(request, **kwargs)
        value.update(ok=False, status='WAIT_STORAGE')
        return value
    result = reporting.run_science_with_report_v1(request, executor=waiting, **options)
    assert result['status'] == 'WAIT_STORAGE'
    assert Path(result['runtime_root']).is_dir()
    assert not options['report_parent'].exists()


def test_failed_report_never_deletes_checkpoint(setup, monkeypatch):
    request, options = setup
    def cannot_report(**kwargs):
        raise ValueError('unavailable reproduction recipe')
    monkeypatch.setattr(reporting, 'build_science_report_v1', cannot_report)
    result = reporting.run_science_with_report_v1(request, executor=executor, **options)
    assert result['status'] == 'CLEANUP_PENDING'
    assert result['ok'] is False
    assert (Path(result['runtime_root']) / 'checkpoint.bin').is_file()


def test_live_owner_queues_cleanup_for_worker(setup, monkeypatch):
    request, options = setup
    def busy(root):
        raise root_retirement.RootRetirementError('live consumer')
    monkeypatch.setattr(root_retirement, '_assert_live_state_clear', busy)
    result = reporting.run_science_with_report_v1(request, executor=executor, **options)
    assert result['status'] == 'CLEANUP_PENDING'
    assert Path(result['report_path']).is_file()
    assert Path(result['runtime_root']).is_dir()
    assert len(list(options['pending_root'].glob('*.json'))) == 1


def test_separate_runs_have_separate_materializations(setup):
    request, options = setup
    seen = []
    def paused(request, **kwargs):
        seen.append(kwargs['materialization_parent'])
        return executor(request, **kwargs) | {'status': 'PAUSED', 'ok': False}
    for _ in range(2):
        reporting.run_science_with_report_v1(request, executor=paused, **options)
    assert len(set(seen)) == 2
    assert all(p.is_dir() for p in seen)


def test_missing_resolved_configuration_preserves_run(setup):
    request, options = setup
    def broken(request, **kwargs):
        result = executor(request, **kwargs)
        (Path(result['materialization_root']) / 'science_request_plan_receipt.json').write_text('{"resolved_inputs":{}}')
        return result
    result = reporting.run_science_with_report_v1(request, executor=broken, **options)
    assert result['status'] == 'CLEANUP_PENDING'
    assert 'missing resolved reproduction input' in result['cleanup_reason']
    assert Path(result['runtime_root']).exists()


def test_public_science_command_uses_report_owner(monkeypatch, tmp_path):
    from eiel_composition import science_instrument_cli as cli
    from eiel_composition import science_request_v1_front_door as front
    calls = []
    monkeypatch.setattr(front, 'validate_science_request_v1', lambda path: {'ok': True})
    def owned(path, **kwargs):
        calls.append((path, kwargs))
        return {'ok': True, 'status': 'COMPLETED', 'report_path': '/reports/run.md'}
    monkeypatch.setattr(reporting, 'run_science_with_report_v1', owned)
    assert cli.main(['run', str(tmp_path / 'request.json'), '--compact-reducer-workers', '2']) == 0
    assert len(calls) == 1
    assert calls[0][1]['compact_reducer_workers'] == 2


def test_invalid_cli_request_never_allocates_report_owner(monkeypatch, tmp_path):
    from eiel_composition import science_instrument_cli as cli
    from eiel_composition import science_request_v1_front_door as front
    monkeypatch.setattr(front, 'validate_science_request_v1', lambda path: {'ok': False, 'status': 'INVALID'})
    monkeypatch.setattr(reporting, 'run_science_with_report_v1', lambda *a, **kw: pytest.fail('owner allocated for invalid request'))
    assert cli.main(['run', str(tmp_path / 'request.json')]) == 2
