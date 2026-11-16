import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from scripts.build_drive_trial_notebook import ARCHIVE_BYTES, ARCHIVE_SHA256
from scripts.package_small_codebook_architecture import notebook
import test_drive_trial_launcher as trial


def sha(data):
    return hashlib.sha256(data).hexdigest()


def environment(monkeypatch, tmp_path, bad_member=False, unsafe=False):
    drive = tmp_path/'content/drive/MyDrive'; drive.mkdir(parents=True)
    base = drive/'genie-baseline-v1-v9.zip'
    with zipfile.ZipFile(base, 'w') as archive:
        archive.writestr('genie_diep/model.py', b'original')
        archive.writestr('data/frozen.npz', b'original data')
        archive.writestr('bundle-sha256.json', json.dumps({
            'genie_diep/model.py': sha(b'original'), 'data/frozen.npz': sha(b'original data')}))
    payloads = {'genie_diep/model.py': b'revised', 'runs/private-initial.pt': b'fresh untrained state'}
    if unsafe: payloads['../../escape.py'] = b'unsafe'
    hashes = {n: sha(v) for n, v in payloads.items()}
    if bad_member: hashes['genie_diep/model.py'] = '0'*64
    overlay = drive/'new-overlay.zip'
    with zipfile.ZipFile(overlay, 'w') as archive:
        for n, v in payloads.items(): archive.writestr(n, v)
        archive.writestr('overlay-sha256.json', json.dumps(hashes))
    manifest = dict(overlay_name=overlay.name, overlay_bytes=overlay.stat().st_size,
        overlay_sha256=sha(overlay.read_bytes()), files=hashes, plan_sha256='a'*64)
    monkeypatch.setattr(trial, 'notebook', lambda: notebook(manifest))
    colab, run, scope, code = trial.launcher_environment(monkeypatch, tmp_path)
    scope['RESUME_RUN_DIRECTORY'] = ''
    code = code.replace(ARCHIVE_SHA256, sha(base.read_bytes())).replace(str(ARCHIVE_BYTES), str(base.stat().st_size))
    return colab, run, scope, code


def test_disabled_configuration_and_shared_watchdog_precede_drive_access():
    document = notebook(dict(overlay_name='private.zip', overlay_bytes=1, overlay_sha256='a'*64, files={}, plan_sha256='b'*64))
    scope = {}; exec(''.join(document['cells'][1]['source']), scope)
    assert scope['AUTHORIZED_TO_RUN'] is False
    code = ''.join(document['cells'][2]['source'])
    assert code.index('threading.Thread(target=watchdog') < code.index("drive.mount('/content/drive')")
    assert 'min(14400, 4 / DISPLAYED_UNITS_PER_HOUR' in code
    assert "'maximum_compute_units': 4, 'maximum_allocated_seconds': 14400" in code


def test_proposed_bound_limits_drive_watchdog_receipt_and_heading_without_authorization():
    document=notebook(dict(overlay_name='private.zip',overlay_bytes=1,overlay_sha256='a'*64,
        files={},plan_sha256='b'*64,maximum_compute_units=4.5,maximum_allocated_seconds=14400,
        minimum_remaining_units=50))
    scope={};exec(''.join(document['cells'][1]['source']),scope)
    assert scope['AUTHORIZED_TO_RUN'] is False
    code=''.join(document['cells'][2]['source'])
    assert 'min(14400, 4.5 / DISPLAYED_UNITS_PER_HOUR' in code
    assert "'maximum_compute_units': 4.5, 'maximum_allocated_seconds': 14400" in code
    assert 'Shared hard limits: 4.5 compute units / 4 allocated hours' in ''.join(document['cells'][0]['source'])
    assert 'min(14400, 4 / DISPLAYED_UNITS_PER_HOUR' not in code


def test_verified_overlay_executes_sequential_runner_and_always_releases(monkeypatch, tmp_path):
    colab, run, scope, code = environment(monkeypatch, tmp_path)
    def execute(command, **kwargs):
        if 'pip' in command: return
        assert 'scripts.run_small_codebook_architecture' in command and '--authorized' in command
        assert '--resume' not in command
        out = Path(command[command.index('--out')+1]); out.mkdir()
        (out/'experiment-report.json').write_text(json.dumps(dict(status='completed')))
    run.side_effect = execute
    exec(compile(code, 'launcher', 'exec'), scope)
    colab.runtime.unassign.assert_called_once()
    assert next(tmp_path.rglob('data/frozen.npz')).read_bytes() == b'original data'
    assert next(tmp_path.rglob('genie_diep/model.py')).read_bytes() == b'revised'
    assert json.loads(next(tmp_path.rglob('launch-status.json')).read_text())['status'] == 'completed'


@pytest.mark.parametrize('bad_member,unsafe,message', [(True, False, 'member changed'), (False, True, 'Unsafe overlay')])
def test_bad_overlay_never_writes_sources_and_releases(monkeypatch, tmp_path, bad_member, unsafe, message):
    colab, run, scope, code = environment(monkeypatch, tmp_path, bad_member, unsafe)
    with pytest.raises(AssertionError, match=message): exec(compile(code, 'launcher', 'exec'), scope)
    run.assert_not_called(); colab.runtime.unassign.assert_called_once()
    assert next(tmp_path.rglob('genie_diep/model.py')).read_bytes() == b'original'


def test_invalid_live_accounting_releases_before_drive_access(monkeypatch, tmp_path):
    colab, run, scope, code = environment(monkeypatch, tmp_path)
    scope['LIVE_BALANCE'] = float('nan')
    with pytest.raises(AssertionError): exec(compile(code, 'launcher', 'exec'), scope)
    colab.drive.mount.assert_not_called(); run.assert_not_called(); colab.runtime.unassign.assert_called_once()


def test_subprocess_timeout_preserves_failure_and_releases(monkeypatch, tmp_path):
    colab, run, scope, code = environment(monkeypatch, tmp_path)
    def execute(command, **kwargs):
        if 'pip' in command: return
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    run.side_effect = execute
    with pytest.raises(subprocess.TimeoutExpired): exec(compile(code, 'launcher', 'exec'), scope)
    colab.runtime.unassign.assert_called_once()
    assert 'TimeoutExpired' in json.loads(next(tmp_path.rglob('launch-status.json')).read_text())['error']
