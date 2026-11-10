"""Exercise budget, immutable-data overlay validation and cleanup without a GPU."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from scripts.build_drive_trial_notebook import ARCHIVE_BYTES, ARCHIVE_SHA256
from scripts.build_recovery_verification_notebook import REQUIRED_MEMBERS, build, notebook
import test_drive_trial_launcher as trial


def sha(content):
    return hashlib.sha256(content).hexdigest()


def fixture_manifest():
    return dict(overlay_name='genie-recovery-overlay-v1.zip', overlay_sha256='a'*64,
                overlay_bytes=100, expected_checkpoint_sha256='b'*64,
                expected_implementation_sha256='c'*64, expected_spec_sha256='d'*64,
                expected_model_sha256='e'*64)


def environment(monkeypatch, tmp_path, extra=None, wrong_member_hash=False, missing=None):
    drive = tmp_path/'content/drive/MyDrive'
    drive.mkdir(parents=True)
    base = drive/'genie-baseline-v1-v9.zip'
    with zipfile.ZipFile(base, 'w') as archive:
        archive.writestr('data/frozen.npz', b'frozen dataset')
        archive.writestr('genie_diep/model.py', b'original source')
        archive.writestr('bundle-sha256.json', json.dumps({
            'data/frozen.npz': sha(b'frozen dataset'), 'genie_diep/model.py': sha(b'original source')}))
    payloads = {name: ('# '+name+'\n').encode() for name in REQUIRED_MEMBERS}
    payloads['genie_diep/model.py'] = b'# revised model\r\n'
    if extra:
        payloads.update(extra)
    if missing:
        payloads.pop(missing)
    hashes = {name: sha(payload) for name, payload in payloads.items()}
    if wrong_member_hash:
        hashes['genie_diep/cohort.py'] = '0'*64
    overlay = drive/'genie-recovery-overlay-v1.zip'
    with zipfile.ZipFile(overlay, 'w') as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
        archive.writestr('overlay-sha256.json', json.dumps(hashes))
    manifest = dict(fixture_manifest(), overlay_sha256=sha(overlay.read_bytes()), overlay_bytes=overlay.stat().st_size,
                    expected_checkpoint_sha256=hashes['recovery-inputs/last.pt'],
                    expected_spec_sha256=hashes['configs/training-baseline-v1.json'],
                    expected_model_sha256=sha(payloads['genie_diep/model.py'].replace(b'\r\n', b'\n')))
    monkeypatch.setattr(trial, 'notebook', lambda: notebook(manifest))
    colab, run, scope, code = trial.launcher_environment(monkeypatch, tmp_path)
    # Bind only this test's tiny base archive; production constants remain unchanged.
    code = code.replace(ARCHIVE_SHA256, sha(base.read_bytes())).replace(str(ARCHIVE_BYTES), str(base.stat().st_size))
    return colab, run, scope, code, manifest


def test_success_applies_only_verified_overlay_and_releases(monkeypatch, tmp_path):
    colab, run, scope, code, manifest = environment(monkeypatch, tmp_path)
    def execute(command, **kwargs):
        if 'pip' in command:
            return
        assert 'scripts.gpu_recovery_verification' in command
        assert command[command.index('--updates')+1] == '300'
        for key in ('expected_checkpoint_sha256', 'expected_implementation_sha256', 'expected_spec_sha256', 'expected_model_sha256'):
            assert command[command.index('--'+key.replace('_', '-'))+1] == manifest[key]
        output = Path(command[command.index('--out')+1])
        output.mkdir(parents=True)
        (output/'verification-report.json').write_text(json.dumps(dict(status='passed')))
    run.side_effect = execute
    exec(compile(code, 'launcher', 'exec'), scope)
    colab.runtime.unassign.assert_called_once()
    root = tmp_path/'content/genie-recovery-verification-v1'
    assert (root/'data/frozen.npz').read_bytes() == b'frozen dataset'
    assert (root/'genie_diep/model.py').read_bytes() == b'# revised model\r\n'
    assert (tmp_path/'content/drive/MyDrive/genie-baseline-v1-v9.zip').is_file()
    status = json.loads(next(tmp_path.rglob('launch-status.json')).read_text())
    assert status['status'] == 'passed' and status['overlay_applied']
    assert status['training_stage_promotion'] is False


@pytest.mark.parametrize('extra,wrong_hash,missing,message', [
    ({'data/frozen.npz': b'changed'}, False, None, 'not allowed'),
    ({'../escape.py': b'escape'}, False, None, 'not allowed'),
    (None, True, None, 'member hash mismatch'),
    (None, False, 'genie_diep/gpu_precision.py', 'Missing required'),
])
def test_bad_overlay_never_changes_source_or_data(monkeypatch, tmp_path, extra, wrong_hash, missing, message):
    colab, run, scope, code, _ = environment(monkeypatch, tmp_path, extra, wrong_hash, missing)
    with pytest.raises(AssertionError, match=message):
        exec(compile(code, 'launcher', 'exec'), scope)
    run.assert_not_called()
    colab.runtime.unassign.assert_called_once()
    root = tmp_path/'content/genie-recovery-verification-v1'
    assert (root/'data/frozen.npz').read_bytes() == b'frozen dataset'
    assert (root/'genie_diep/model.py').read_bytes() == b'original source'


def test_subprocess_timeout_saves_failure_and_releases(monkeypatch, tmp_path):
    colab, run, scope, code, _ = environment(monkeypatch, tmp_path)
    def execute(command, **kwargs):
        if 'pip' not in command:
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    run.side_effect = execute
    with pytest.raises(subprocess.TimeoutExpired):
        exec(compile(code, 'launcher', 'exec'), scope)
    colab.runtime.unassign.assert_called_once()
    status = json.loads(next(tmp_path.rglob('launch-status.json')).read_text())
    assert status['status'] == 'failed' and 'TimeoutExpired' in status['error']


@pytest.mark.parametrize('seconds,units,rate,balance', [(3500,0,1,100), (0,1.99,1,100), (0,0,100,100), (0,0,1,50.01)])
def test_budget_exhaustion_releases_before_mount(monkeypatch, tmp_path, seconds, units, rate, balance):
    monkeypatch.setattr(trial, 'notebook', lambda: notebook(fixture_manifest()))
    colab, run, scope, code = trial.launcher_environment(monkeypatch, tmp_path)
    scope.update(PRIOR_ALLOCATED_SECONDS=seconds, PRIOR_COMPUTE_UNITS=units,
                 DISPLAYED_UNITS_PER_HOUR=rate, LIVE_BALANCE=balance)
    with pytest.raises(AssertionError, match='Insufficient budget'):
        exec(compile(code, 'launcher', 'exec'), scope)
    colab.drive.mount.assert_not_called()
    colab.runtime.unassign.assert_called_once()
    run.assert_not_called()


def test_export_disabled_and_never_overwrites(tmp_path):
    manifest = tmp_path/'overlay.json'
    manifest.write_text(json.dumps(fixture_manifest()))
    output = tmp_path/'launcher'
    result = build(output, manifest)
    document = json.loads((output/result['notebook']).read_text())
    scope = {}
    exec(''.join(document['cells'][1]['source']), scope)
    assert scope['AUTHORIZED_TO_RUN'] is False
    assert result['maximum_optimizer_calls'] == 300
    with pytest.raises(FileExistsError):
        build(output, manifest)


@pytest.mark.parametrize('name', ['../overlay.zip', "bad'name.zip", 'genie-baseline-v1-v9.zip'])
def test_overlay_filename_validation(name):
    with pytest.raises(ValueError):
        notebook(dict(fixture_manifest(), overlay_name=name))
