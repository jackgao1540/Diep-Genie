"""Launcher safety checks only; these do not run or validate a GPU model."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.build_drive_trial_notebook import build, notebook


def test_versioned_archive_is_bound_without_old_archive_references():
    document=notebook('genie-baseline-v2-v9.zip','a'*64,12345)
    source=''.join(document['cells'][2]['source'])
    compile(source,'versioned-launcher','exec')
    assert 'genie-baseline-v2-v9.zip' in source and "'a"+('a'*63)+"'" in source
    assert '12345' in source and 'baseline-v1-v9' not in source
    assert 'AUTHORIZED_TO_RUN = False' in ''.join(document['cells'][1]['source'])


@pytest.mark.parametrize('name',['../bundle.zip',"bad\'name.zip",'bundle.zip/else','bundle.py'])
def test_archive_name_cannot_escape_or_inject_code(name):
    with pytest.raises(ValueError,match='filename'):notebook(name,'a'*64,100)


def launcher_environment(monkeypatch, tmp_path):
    colab = ModuleType('google.colab')
    colab.drive = SimpleNamespace(mount=Mock())
    colab.runtime = SimpleNamespace(unassign=Mock())
    google = ModuleType('google')
    google.colab = colab
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    # Redirect only the notebook's filesystem into the test directory.
    monkeypatch.setitem(sys.modules, 'pathlib', SimpleNamespace(Path=lambda p: tmp_path/str(p).lstrip('/')))
    import threading
    import subprocess
    monkeypatch.setattr(threading, 'Thread', Mock())
    run = Mock(side_effect=AssertionError('Setup must not execute after validation failure'))
    monkeypatch.setattr(subprocess, 'run', run)
    scope = dict(AUTHORIZED_TO_RUN=True, LIVE_BALANCE=100.0, DISPLAYED_UNITS_PER_HOUR=2.0,
                 PRIOR_ALLOCATED_SECONDS=0.0, PRIOR_COMPUTE_UNITS=0.0,
                 ALLOCATION_START_UTC=datetime.now(timezone.utc).isoformat())
    code = ''.join(notebook()['cells'][2]['source'])
    return colab, run, scope, code


def test_incomplete_upload_stops_before_install_and_releases_runtime(monkeypatch, tmp_path):
    source = tmp_path/'content/drive/MyDrive/genie-baseline-v1-v9.zip'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'incomplete upload')
    colab, run, scope, code = launcher_environment(monkeypatch, tmp_path)
    with pytest.raises(AssertionError, match='Archive size mismatch'):
        exec(compile(code, 'launcher', 'exec'), scope)
    run.assert_not_called()
    colab.runtime.unassign.assert_called_once()
    report = json.loads(next(tmp_path.rglob('launch-status.json')).read_text())
    assert report['status'] == 'failed'
    assert report['training_stage_promotion'] is False


def test_expired_budget_releases_runtime_without_mount(monkeypatch, tmp_path):
    colab, run, scope, code = launcher_environment(monkeypatch, tmp_path)
    scope['ALLOCATION_START_UTC'] = datetime.fromtimestamp(time.time()-3601, timezone.utc).isoformat()
    with pytest.raises(AssertionError, match='Insufficient budget'):
        exec(compile(code, 'launcher', 'exec'), scope)
    colab.drive.mount.assert_not_called()
    run.assert_not_called()
    colab.runtime.unassign.assert_called_once()


def test_default_launcher_disabled_and_export_cannot_overwrite(tmp_path):
    output = tmp_path/'launch-v1'
    build(output)
    exported = json.loads((output/'first-gpu-trial-drive-v2.ipynb').read_text())
    scope = {}
    exec(''.join(exported['cells'][1]['source']), scope)
    assert scope['AUTHORIZED_TO_RUN'] is False
    with pytest.raises(FileExistsError):
        build(output)


@pytest.mark.parametrize('seconds,units,rate', [(3400, 0, 1), (0, 1.99, 1), (300, 0.1, 100)])
def test_prior_debits_or_live_rate_prevent_overbudget_setup(monkeypatch, tmp_path, seconds, units, rate):
    colab, run, scope, code = launcher_environment(monkeypatch, tmp_path)
    scope.update(PRIOR_ALLOCATED_SECONDS=seconds, PRIOR_COMPUTE_UNITS=units,
                 DISPLAYED_UNITS_PER_HOUR=rate)
    with pytest.raises(AssertionError, match='Insufficient budget'):
        exec(compile(code, 'launcher', 'exec'), scope)
    colab.drive.mount.assert_not_called()
    run.assert_not_called()
    colab.runtime.unassign.assert_called_once()
    remaining = scope['deadline'] - scope['allocation_start']
    assert remaining <= 3600 - seconds + 1e-6
    assert remaining * rate / 3600 <= 2 - units + 1e-6
