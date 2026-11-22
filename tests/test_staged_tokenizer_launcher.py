"""Exercise the private pilot boundary with mocked Colab; no GPU is allocated."""
import json
import hashlib
from pathlib import Path
import subprocess
import time
import zipfile

import pytest

from scripts.package_staged_tokenizer import notebook, full_notebook, package
import test_small_codebook_launcher as prior


def environment(monkeypatch, tmp_path, **kwargs):
    monkeypatch.setattr(prior, 'notebook', lambda manifest: notebook(dict(manifest,
        plan_path='configs/staged-tokenizer-run-v1.json')))
    colab, run, scope, code = prior.environment(monkeypatch, tmp_path, **kwargs)
    return colab, run, scope, code


def test_pilot_only_and_guard_before_mount():
    doc = notebook(dict(overlay_name='private.zip', overlay_bytes=1,
        overlay_sha256='a'*64, files={}, plan_sha256='b'*64,
        plan_path='configs/staged-tokenizer-run-v1.json'))
    scope = {}; exec(''.join(doc['cells'][1]['source']), scope)
    assert scope['AUTHORIZED_TO_RUN'] is False
    code = ''.join(doc['cells'][2]['source'])
    assert code.index('threading.Thread(target=watchdog') < code.index('drive.mount(')
    assert "'--mode', 'pilot'" in code and '--full-budget' not in code
    assert 'min(2700 - PRIOR_ALLOCATED_SECONDS,' in code
    assert '(1 - PRIOR_COMPUTE_UNITS) / DISPLAYED_UNITS_PER_HOUR' in code
    assert "'maximum_compute_units': 1, 'maximum_allocated_seconds': 2700" in code


def test_distinct_full_launcher_requires_receipt_and_preserves_total_caps():
    manifest = dict(overlay_name='private.zip', overlay_bytes=1, overlay_sha256='a'*64,
        files={}, plan_sha256='b'*64, plan_path='configs/staged-tokenizer-run-v1.json')
    with pytest.raises(ValueError, match='receipt'): full_notebook(manifest)
    manifest.update(mode='full', full_budget_path='runs/private/full-budget.json')
    doc = full_notebook(manifest)
    scope = {}; exec(''.join(doc['cells'][1]['source']), scope)
    assert scope['AUTHORIZED_TO_RUN'] is False
    code = ''.join(doc['cells'][2]['source'])
    assert "'--mode', 'full', '--full-budget'" in code
    assert "'--mode', 'pilot'" not in code
    assert 'min(93600 - PRIOR_ALLOCATED_SECONDS,' in code
    assert '(30 - PRIOR_COMPUTE_UNITS) / DISPLAYED_UNITS_PER_HOUR' in code
    assert "'maximum_compute_units': 30, 'maximum_allocated_seconds': 93600" in code
    assert '/Diep-Genie/staged-tokenizer-full-v1' in code
    assert '/Diep-Genie/staged-tokenizer-run-v1' not in code
    assert code.index('threading.Thread(target=watchdog') < code.index('drive.mount(')


def test_full_package_binds_raw_pilot_bytes_and_debit_floors_without_weights(tmp_path):
    root = Path(__file__).resolve().parents[1]
    pilot = dict(status='pilot_complete', identity=dict(mode='pilot', local_check=False,
        plan_sha256=hashlib.sha256((root/'configs/staged-tokenizer-run-v1.json').read_bytes()).hexdigest()),
        spent_seconds=10, spent_units=.01)
    report = tmp_path/'private-pilot.json'; report.write_text(json.dumps(pilot)+'\n')
    with pytest.raises(ValueError, match='debits'):
        package(root, tmp_path/'rejected', pilot_report=report, prior_seconds=9, prior_units=.01)
    assert not (tmp_path/'rejected').exists()
    output = tmp_path/'full-export'
    manifest = package(root, output, pilot_report=report, prior_seconds=20, prior_units=.02)
    assert manifest['mode'] == 'full' and manifest['contains_checkpoints'] is False
    assert manifest['maximum_compute_units'] == 30 and manifest['maximum_allocated_seconds'] == 93600
    with zipfile.ZipFile(output/manifest['overlay_name']) as archive:
        receipt = json.loads(archive.read(manifest['full_budget_path']))
        assert receipt['minimum_prior_seconds'] == 20 and receipt['minimum_prior_units'] == .02
        assert archive.read(receipt['pilot']['report_path']) == report.read_bytes()
        assert receipt['pilot']['sha256'] == hashlib.sha256(report.read_bytes()).hexdigest()
        assert all(not name.endswith(('.pt','.npz','.mp4')) for name in archive.namelist())
    with pytest.raises(FileExistsError): package(root, output)


@pytest.mark.parametrize('changes', [dict(AUTHORIZED_TO_RUN=False), dict(LIVE_BALANCE=float('nan')),
    dict(DISPLAYED_UNITS_PER_HOUR=0), dict(PRIOR_COMPUTE_UNITS=1),
    dict(PRIOR_ALLOCATED_SECONDS=2700), dict(ALLOCATION_START_UTC='bad')])
def test_invalid_or_exhausted_start_releases_before_mount(monkeypatch, tmp_path, changes):
    colab, run, scope, code = environment(monkeypatch, tmp_path)
    scope.update(changes)
    with pytest.raises((ValueError, AssertionError)): exec(compile(code, 'pilot', 'exec'), scope)
    colab.drive.mount.assert_not_called(); run.assert_not_called()
    colab.runtime.unassign.assert_called_once()


def test_verified_overlay_runs_only_pilot_and_counts_prior_debits(monkeypatch, tmp_path):
    colab, run, scope, code = environment(monkeypatch, tmp_path)
    scope.update(PRIOR_ALLOCATED_SECONDS=60, PRIOR_COMPUTE_UNITS=.1)
    def execute(command, **kwargs):
        if 'pip' in command: return
        assert command[command.index('--mode')+1] == 'pilot'
        assert command[command.index('--prior-seconds')+1] == '60'
        assert command[command.index('--prior-units')+1] == '0.1'
        assert float(command[command.index('--allocation-start')+1]) == scope['allocation_start']
        assert '--full-budget' not in command
        # Runner stops/checkpoints 180s before its exact deadline. Shell kill
        # leaves another90s after that, then60s before watchdog unassignment.
        assert scope['stop_at'] == scope['deadline'] - 30
        assert abs(kwargs['timeout'] - (scope['deadline'] - time.time() - 90)) < 1
        assert kwargs['timeout'] > scope['deadline'] - time.time() - 180
    run.side_effect = execute
    exec(compile(code, 'pilot', 'exec'), scope)
    colab.runtime.unassign.assert_called_once()
    report = json.loads(next(tmp_path.rglob('launch-status.json')).read_text())
    assert report['status'] == 'pilot_runner_completed'
    assert report['total_seconds_with_prior_debit'] >= 60
    assert report['total_units_with_prior_debit'] >= .1


@pytest.mark.parametrize('seconds,units', [(800,.01), (1,.4)])
def test_outer_deadline_uses_same_separate_debits_as_runner(monkeypatch, tmp_path, seconds, units):
    colab, run, scope, code = environment(monkeypatch, tmp_path)
    scope.update(PRIOR_ALLOCATED_SECONDS=seconds, PRIOR_COMPUTE_UNITS=units)
    run.side_effect = lambda *args, **kwargs: None
    exec(compile(code, 'pilot', 'exec'), scope)
    expected = min(2700-seconds, (1-units)/scope['DISPLAYED_UNITS_PER_HOUR']*3600,
                   (scope['LIVE_BALANCE']-50)/scope['DISPLAYED_UNITS_PER_HOUR']*3600)
    assert abs(scope['deadline'] - scope['allocation_start'] - expected) < 1e-6
    colab.runtime.unassign.assert_called_once()


def test_mount_failure_has_local_receipt_and_releases(monkeypatch, tmp_path):
    colab, run, scope, code = environment(monkeypatch, tmp_path)
    colab.drive.mount.side_effect = RuntimeError('mount failed')
    with pytest.raises(RuntimeError, match='mount failed'): exec(compile(code, 'pilot', 'exec'), scope)
    colab.runtime.unassign.assert_called_once(); run.assert_not_called()
    report = json.loads(next(tmp_path.rglob('staged-pilot-launch-status.json')).read_text())
    assert report['status'] == 'failed' and 'mount failed' in report['error']


def test_timeout_has_failure_receipt_and_releases(monkeypatch, tmp_path):
    colab, run, scope, code = environment(monkeypatch, tmp_path)
    def execute(command, **kwargs):
        if 'pip' not in command: raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    run.side_effect = execute
    with pytest.raises(subprocess.TimeoutExpired): exec(compile(code, 'pilot', 'exec'), scope)
    colab.runtime.unassign.assert_called_once()
    assert 'TimeoutExpired' in json.loads(next(tmp_path.rglob('launch-status.json')).read_text())['error']


@pytest.mark.parametrize('kwargs', [dict(bad_member=True), dict(unsafe=True)])
def test_bad_overlay_never_executes_and_releases(monkeypatch, tmp_path, kwargs):
    colab, run, scope, code = environment(monkeypatch, tmp_path, **kwargs)
    with pytest.raises(AssertionError): exec(compile(code, 'pilot', 'exec'), scope)
    run.assert_not_called(); colab.runtime.unassign.assert_called_once()
