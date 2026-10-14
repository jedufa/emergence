"""Top-level science ownership and report-only terminal cleanup.

Nested science consumers keep using the composition API. The public operator
CLI owns one fresh bundle, so its materializations cannot be shared by another
run and can be retired together after reduction and aggregation.
"""
from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, Callable, Mapping

from engine.campaigns.artifact_lifecycle import ArtifactLifecycleError, _atomic_json
from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.report_only_closeout import closeout_report_only_run_v1, _write

REPORT_ROOT = Path(public_runtime_path('run-reports'))
OWNER_ROOT = Path(public_runtime_path('science-request-v1/report-only-runs'))
PENDING_ROOT = Path(public_runtime_path('lifecycle/pending'))
MAX_REPORT_BYTES = 1024 * 1024


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + '\n'


def _block(value: Any, language: str = 'json') -> str:
    text = _json(value) if language == 'json' else str(value)
    fence = '`' * max(3, max((len(x) for x in __import__('re').findall(r'`+', text)), default=0) + 1)
    return f'{fence}{language}\n{text}\n{fence}\n'


def _read(path: Path, *, max_bytes: int = MAX_REPORT_BYTES) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ArtifactLifecycleError(f'missing, linked, or oversized report input: {path}')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ArtifactLifecycleError(f'report input must be an object: {path}')
    return value


def _source_context(repo: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True)
    versions = {}
    for name in ('numpy', 'torch', 'PyYAML'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    # Preserve changed source as a reproduction patch, never binary evidence.
    scopes = ('engine', 'eiel_composition', 'emergence_contracts', 'emergence_evidence', 'tools',
              'pyproject.toml', 'setup.cfg', 'requirements.txt')
    patch = git('diff', 'HEAD', '--', *scopes)
    if 'Binary files ' in patch or 'GIT binary patch' in patch:
        raise ArtifactLifecycleError('commit changed binary source inputs before execution')
    untracked = git('ls-files', '--others', '--exclude-standard', '--', *scopes)
    additions = {}
    for name in untracked.splitlines():
        path = repo / name
        if path.is_symlink() or path.stat().st_size > MAX_REPORT_BYTES:
            raise ArtifactLifecycleError(f'unreportable source input: {name}')
        additions[name] = path.read_text()
    value = {'commit': git('rev-parse', 'HEAD').strip(), 'python': platform.python_version(),
             'platform': platform.platform(), 'packages': versions, 'source_patch': patch,
             'additional_source_files': additions,
             'execution_source_scopes': scopes,
             'execution_source_state': 'clean' if not patch and not additions else 'inline_changes'}
    if len(_json(value).encode()) > MAX_REPORT_BYTES // 2:
        raise ArtifactLifecycleError('commit source changes before running: reproduction source exceeds report budget')
    return value


def _summary(value: Any) -> Any:
    """Keep aggregate decisions/metrics, not raw per-world or per-event rows."""
    if isinstance(value, dict):
        return {k: _summary(v) for k, v in value.items()
                if k not in {'worlds', 'world_results', 'world_rows', 'events', 'records', 'artifacts', 'snapshots'}}
    if isinstance(value, list):
        if len(value) > 64:
            return {'row_count': len(value), 'detail': 'raw rows deliberately not retained'}
        return [_summary(x) for x in value]
    return value


def build_science_report_v1(*, owner: Path, request: Mapping[str, Any], result: Mapping[str, Any], source: Mapping[str, Any]) -> str:
    runtime = Path(str(result['runtime_root']))
    materialization = Path(str(result['materialization_root']))
    if owner not in runtime.parents or owner not in materialization.parents:
        raise ArtifactLifecycleError('report-only bundle does not own exact science outputs')
    if result.get('status') != 'COMPLETED' or result.get('ok') is not True:
        raise ArtifactLifecycleError('unfinished science cannot be retired')
    lifecycle = result.get('lifecycle_closeout', {})
    if lifecycle.get('inventory_complete') is not True or lifecycle.get('terminal_status') != 'completed':
        raise ArtifactLifecycleError('science lifecycle is not sealed and complete')
    aggregate = result.get('aggregate')
    results = result.get('host_observation_closeout')
    if isinstance(aggregate, dict) and isinstance(aggregate.get('result'), dict):
        reference = aggregate['result']
        path = Path(reference['path'])
        if runtime not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != reference['file_sha256']:
            raise ArtifactLifecycleError('aggregate report input binding differs')
        results = _summary(_read(path, max_bytes=64 * 1024 * 1024))
    if not isinstance(results, dict) or not results:
        raise ArtifactLifecycleError('completed science lacks aggregate or observation results')
    plan = _read(materialization / 'science_request_plan_receipt.json')
    resolved = {}
    for key in ('base_campaign_plan', 'science_context'):
        ref = plan.get('resolved_inputs', {}).get(key)
        if not isinstance(ref, dict):
            raise ArtifactLifecycleError(f'missing resolved reproduction input: {key}')
        path = Path(ref['path'])
        if materialization not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != ref['file_sha256']:
            raise ArtifactLifecycleError('resolved reproduction input binding differs')
        resolved[key] = _read(path)
    input_files = {}
    # Small initial-condition banks are inputs, not output checkpoints. Keep the
    # exact author-authored values in the recipe when this request uses one.
    bank = request.get('exact_natural_source_state_bank')
    if isinstance(bank, dict) and isinstance(bank.get('path'), str):
        path = Path(bank['path'])
        raw = path.read_bytes()
        if len(raw) > 256 * 1024 or hashlib.sha256(raw).hexdigest() != bank.get('file_sha256'):
            raise ArtifactLifecycleError('initial-condition bank needs a bounded regeneration recipe')
        input_files[bank['path']] = json.loads(raw)
    if request.get('continuation'):
        raise ArtifactLifecycleError('historical-state run needs an authored regeneration report before cleanup')
    report = (
        f"# Science run: {request.get('campaign_id')}\n\n{request.get('question', '')}\n\n"
        '## Inputs\n\nSave this complete authoring request as `request.json`.\n\n' + _block(dict(request)) +
        '\nInitial-condition input files (recreate these paths or rebind their path and digest):\n\n' + _block(input_files) +
        '\n## Settings\n\nSource, software, and hardware identity at launch:\n\n' + _block(dict(source)) +
        '\nResolved plan and observation context used by this execution:\n\n' + _block(resolved) +
        '\n## Results\n\nExecution and aggregation completed. These are the recorded aggregate results; '
        'mechanical completion alone is not a positive scientific conclusion.\n\n' + _block(results) +
        '\n## Limitations\n\nRaw events, evidence packages, temporary files and output checkpoints are not retained. '
        'No new Qualification or inheritance is claimed. Reproduction requires the recorded software and '
        'compatible hardware; bitwise equality is not promised across environments. Embedded historical paths '
        'are provenance labels, not retained output dependencies. Optional interesting checkpoints may be '
        'deleted at any time and are never required by this recipe.\n\n'
        '## Reproduction\n\n'
        f"1. Check out source commit `{source['commit']}` in an isolated checkout. Apply `source_patch` with `git apply` "
        'and recreate `additional_source_files` from the Settings block if present.\n'
        '2. Install the recorded Python/package versions and build the native route with that checkout’s canonical '
        'build procedure. Recreate initial-condition files from Inputs, preserving values and validating digests.\n'
        '3. Use the resolved base campaign plan and science context above to reconstruct the same machine authority '
        'through `eiel science activate-authority`; regenerate native binary bindings on the compatible machine. '
        'Do not reuse deleted output paths or optional checkpoint-cache paths as source inputs.\n'
        '4. Save the Inputs request as `request.json`, then run '
        f"`./tools/bin/eiel science run request.json --compact-reducer-workers {source.get('compact_reducer_workers', 4)} --human`. "
        'It creates a fresh run. Seeds, controls, dimensions, absolute ticks, observation schedules and reducer '
        'settings are in the request. Compare the Results metrics with the same interpretation limits.\n'
        '5. To keep a selected checkpoint on a rerun, explicitly select its absolute tick before execution and '
        'export a self-contained optional copy to `$EIEL_RUNTIME_ROOT/interesting-checkpoints`. This is not a dependency '
        'of this run or recipe.\n'
    )
    if len(report.encode()) > MAX_REPORT_BYTES:
        raise ArtifactLifecycleError('report exceeds bounded text budget; author a compact regeneration recipe')
    return report


def queue_report_cleanup_v1(*, report_path: Path, roots: list[Path], source_commit: str, run_id: str, pending_root: Path = PENDING_ROOT) -> Path:
    core = {'schema_version': 'eiel_report_only_cleanup_pointer_v1', 'run_id': run_id,
            'report_path': str(report_path.absolute()), 'roots': [str(p.absolute()) for p in roots], 'source_commit': source_commit}
    digest = canonical_sha256(core)
    return _atomic_json(pending_root / f'report-only-{digest}.json', {**core, 'pointer_sha256': digest})


def run_science_with_report_v1(request_path: Path, *, executor: Callable[..., dict], repo_root: Path,
                               resume_root: Path | None = None, reducer_recovery_source_root: Path | None = None,
                               compact_reducer_workers: int = 4, owner_parent: Path = OWNER_ROOT,
                               report_parent: Path = REPORT_ROOT, pending_root: Path = PENDING_ROOT) -> dict:
    """Public top-level run: isolated owned outputs, report, automatic cleanup."""
    request = _read(request_path)
    if resume_root is None:
        source = _source_context(repo_root)
        source['compact_reducer_workers'] = compact_reducer_workers
        owner_parent.mkdir(parents=True, exist_ok=True)
        owner = Path(tempfile.mkdtemp(prefix='run-', dir=owner_parent))
        _atomic_json(owner / 'report-owner.json', {
            'schema_version': 'eiel_report_only_run_owner_v1',
            'runtime_root': str(owner), 'source': source, 'request': request,
        })
    else:
        resume = resume_root.expanduser().resolve()
        owner = resume.parent.parent
        marker = owner / 'report-owner.json'
        if not marker.is_file():
            # Legacy continuations preserve their existing binding. The operator
            # owns report-only closeout through storage finish-run afterwards.
            result = executor(request_path, resume_root=resume_root,
                              reducer_recovery_source_root=reducer_recovery_source_root,
                              compact_reducer_workers=compact_reducer_workers)
            if result.get('status') == 'COMPLETED':
                return {**result, 'ok': False, 'status': 'CLEANUP_PENDING',
                        'cleanup_reason': 'Legacy run requires a self-contained report and storage finish-run.'}
            return result
        saved = _read(marker)
        if saved['request'] != request:
            raise ArtifactLifecycleError('resumed report owner request differs')
        source = saved['source']
    result = executor(request_path, repo_root=repo_root, materialization_parent=owner / 'materializations',
                      runtime_parent=owner / 'runs', resume_root=resume_root,
                      reducer_recovery_source_root=reducer_recovery_source_root,
                      compact_reducer_workers=compact_reducer_workers)
    if result.get('status') != 'COMPLETED' or result.get('ok') is not True:
        return result
    report_path = report_parent / f'{owner.name}.md'
    try:
        report = build_science_report_v1(owner=owner, request=request, result=result, source=source)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.exists():
            if report_path.read_text() != report:
                raise ArtifactLifecycleError('existing report differs; resume pending cleanup through its queue')
        else:
            _write(report_path, report)
        queue_report_cleanup_v1(report_path=report_path, roots=[owner], source_commit=source['commit'],
                                run_id=str(request['campaign_id']), pending_root=pending_root)
        closed = closeout_report_only_run_v1(report_path=report_path, report_markdown=report, roots=[owner],
                                            source_commit=source['commit'], run_id=str(request['campaign_id']), apply=True)
        # The worker consumes this pointer idempotently and removes obsolete
        # nested lifecycle pointers; the report is the sole permanent output.
        return {'ok': True, 'status': 'COMPLETED', 'campaign_id': request['campaign_id'],
                'report_path': str(report_path), 'cleanup': closed['cleanup'], 'execution_started': True}
    except (ArtifactLifecycleError, OSError, ValueError) as error:
        return {**result, 'ok': False, 'status': 'CLEANUP_PENDING', 'report_path': str(report_path),
                'cleanup_reason': str(error), 'science_execution_completed': True}
