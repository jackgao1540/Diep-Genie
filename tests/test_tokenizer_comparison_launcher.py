"""Exercise source-only packaging and shared allocation caps without a GPU."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from scripts.build_drive_trial_notebook import ARCHIVE_BYTES, ARCHIVE_SHA256
from scripts.build_tokenizer_comparison_notebook import BOUND_FILES, HASH_FIELDS, REQUIRED_MEMBERS, build, notebook
import scripts.package_tokenizer_comparison as package
import test_drive_trial_launcher as trial


def sha(value):
    return hashlib.sha256(value).hexdigest()


def implementation(payloads):
    digest = hashlib.sha256()
    for name in ['baseline.py', 'cohort.py', 'model.py', 'baseline_evaluation.py']:
        digest.update(name.encode())
        digest.update(payloads['genie_diep/' + name].replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def fixture_manifest():
    return dict(overlay_name='genie-tokenizer-comparison-v1.zip', overlay_sha256='a'*64,
                overlay_bytes=100, **{field: 'b'*64 for field in HASH_FIELDS})


def environment(monkeypatch, tmp_path, extra=None, missing=None, wrong_hash=False, wrong_binding=False):
    drive = tmp_path/'content/drive/MyDrive'
    drive.mkdir(parents=True)
    base = drive/'genie-baseline-v1-v9.zip'
    with zipfile.ZipFile(base, 'w') as archive:
        archive.writestr('data/frozen.npz', b'frozen dataset')
        archive.writestr('genie_diep/model.py', b'original source')
        archive.writestr('bundle-sha256.json', json.dumps({
            'data/frozen.npz': sha(b'frozen dataset'), 'genie_diep/model.py': sha(b'original source')}))
    payloads = {name: ('# ' + name + '\n').encode() for name in REQUIRED_MEMBERS}
    payloads['genie_diep/model.py'] = b'# revised source\r\n'
    if extra:
        payloads.update(extra)
    if missing:
        payloads.pop(missing)
    hashes = {name: sha(payload) for name, payload in payloads.items()}
    manifest = fixture_manifest()
    manifest.update({field: hashes.get(name, 'f'*64) for field, name in BOUND_FILES.items()})
    manifest['expected_implementation_sha256'] = implementation(payloads)
    if wrong_binding:
        manifest['expected_experiment_sha256'] = '0'*64
    if wrong_hash:
        hashes['genie_diep/cohort.py'] = '0'*64
    overlay = drive/manifest['overlay_name']
    with zipfile.ZipFile(overlay, 'w') as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
        archive.writestr('overlay-sha256.json', json.dumps(hashes))
    manifest.update(overlay_sha256=sha(overlay.read_bytes()), overlay_bytes=overlay.stat().st_size)
    monkeypatch.setattr(trial, 'notebook', lambda: notebook(manifest))
    colab, run, scope, code = trial.launcher_environment(monkeypatch, tmp_path)
    code = code.replace(ARCHIVE_SHA256, sha(base.read_bytes())).replace(str(ARCHIVE_BYTES), str(base.stat().st_size))
    return colab, run, scope, code


@pytest.mark.parametrize('outcome', ['completed', 'budget_exhausted'])
def test_success_or_budget_stop_preserves_data_and_releases(monkeypatch, tmp_path, outcome):
    colab, run, scope, code = environment(monkeypatch, tmp_path)
    scope.update(PRIOR_ALLOCATED_SECONDS=45, PRIOR_COMPUTE_UNITS=.02)
    def execute(command, **kwargs):
        if 'pip' in command:
            return
        assert 'scripts.run_tokenizer_comparison' in command and '--authorized' in command
        assert command[command.index('--device')+1] == 'cuda'
        assert command[command.index('--baseline-spec')+1] == 'configs/tokenizer-comparison-baseline-v1.json'
        assert command[command.index('--detail-spec')+1] == 'configs/tokenizer-comparison-detail-v1.json'
        assert command[command.index('--experiment')+1] == 'configs/tokenizer-comparison-v1.json'
        assert float(command[command.index('--allocation-start')+1]) == scope['accounting_start']
        assert scope['accounting_start'] <= scope['allocation_start'] - 45
        output = Path(command[command.index('--out')+1])
        output.mkdir(parents=True)
        (output/'comparison-report.json').write_text(json.dumps(dict(status=outcome)))
    run.side_effect = execute
    exec(compile(code, 'launcher', 'exec'), scope)
    colab.runtime.unassign.assert_called_once()
    root = tmp_path/'content/genie-tokenizer-comparison-v1'
    assert (root/'data/frozen.npz').read_bytes() == b'frozen dataset'
    assert (root/'genie_diep/model.py').read_bytes() == b'# revised source\r\n'
    status = json.loads(next(tmp_path.rglob('launch-status.json')).read_text())
    assert status['status'] == outcome and status['overlay_applied']
    assert status['maximum_optimizer_calls_per_arm'] == 1000 and status['budget_shared_by_both_arms']
    assert status['maximum_unique_updates_per_arm'] == status['final_common_checkpoint_step'] == 999
    assert status['quality_gates'] == 'unmeasured' and not status['training_stage_promotion']


@pytest.mark.parametrize('extra,missing,wrong_hash,wrong_binding,message', [
    ({'weights/last.pt': b'private weights'}, None, False, False, 'exactly'),
    ({'data/frozen.npz': b'changed data'}, None, False, False, 'exactly'),
    ({'../escape.py': b'escape'}, None, False, False, 'exactly'),
    (None, 'genie_diep/detail_loss.py', False, False, 'exactly'),
    (None, None, True, False, 'member hash mismatch'),
    (None, None, False, True, 'Input binding mismatch'),
])
def test_bad_overlay_fails_before_any_source_write(monkeypatch, tmp_path, extra, missing, wrong_hash, wrong_binding, message):
    colab, run, scope, code = environment(monkeypatch, tmp_path, extra, missing, wrong_hash, wrong_binding)
    with pytest.raises(AssertionError, match=message):
        exec(compile(code, 'launcher', 'exec'), scope)
    run.assert_not_called()
    colab.runtime.unassign.assert_called_once()
    root = tmp_path/'content/genie-tokenizer-comparison-v1'
    assert (root/'genie_diep/model.py').read_bytes() == b'original source'
    assert (root/'data/frozen.npz').read_bytes() == b'frozen dataset'


def test_timeout_preserves_failure_status_and_releases(monkeypatch, tmp_path):
    colab, run, scope, code = environment(monkeypatch, tmp_path)
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
def test_shared_budget_expiry_releases_before_mount(monkeypatch, tmp_path, seconds, units, rate, balance):
    monkeypatch.setattr(trial, 'notebook', lambda: notebook(fixture_manifest()))
    colab, run, scope, code = trial.launcher_environment(monkeypatch, tmp_path)
    scope.update(PRIOR_ALLOCATED_SECONDS=seconds, PRIOR_COMPUTE_UNITS=units,
                 DISPLAYED_UNITS_PER_HOUR=rate, LIVE_BALANCE=balance)
    with pytest.raises(AssertionError, match='Insufficient budget'):
        exec(compile(code, 'launcher', 'exec'), scope)
    colab.runtime.unassign.assert_called_once()
    colab.drive.mount.assert_not_called()
    run.assert_not_called()


def test_invalid_live_accounting_also_releases(monkeypatch, tmp_path):
    monkeypatch.setattr(trial, 'notebook', lambda: notebook(fixture_manifest()))
    colab, run, scope, code = trial.launcher_environment(monkeypatch, tmp_path)
    scope['LIVE_BALANCE'] = float('nan')
    with pytest.raises(AssertionError):
        exec(compile(code, 'launcher', 'exec'), scope)
    colab.runtime.unassign.assert_called_once()
    colab.drive.mount.assert_not_called()
    run.assert_not_called()


def test_disabled_export_cannot_overwrite_and_has_one_shared_watchdog(tmp_path):
    manifest = tmp_path/'overlay.json'
    manifest.write_text(json.dumps(fixture_manifest()))
    result = build(tmp_path/'launcher', manifest)
    document = json.loads((tmp_path/'launcher'/result['notebook']).read_text())
    scope = {}
    exec(''.join(document['cells'][1]['source']), scope)
    source = ''.join(document['cells'][2]['source'])
    assert scope['AUTHORIZED_TO_RUN'] is False
    assert source.count('threading.Thread(target=watchdog') == 1
    assert source.index('threading.Thread(target=watchdog') < source.index("drive.mount('/content/drive')")
    assert 'stop_at-time.time()-180' in source
    assert result['budget_shared_by_both_arms'] and result['maximum_optimizer_calls_per_arm'] == 1000
    assert result['maximum_unique_updates_per_arm'] == result['final_common_checkpoint_step'] == 999
    assert 'including\none exact-resume replay' in ''.join(document['cells'][0]['source'])
    assert not result['contains_existing_weights']
    with pytest.raises(FileExistsError):
        build(tmp_path/'launcher', manifest)


def test_packager_refuses_missing_inputs_before_creating_export(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(package, 'validate_readiness', lambda *args: None)
    monkeypatch.setattr(package, 'validate_comparison', lambda *args: None)
    output = tmp_path/'export'
    with pytest.raises(FileNotFoundError):
        package.run(output)
    assert not output.exists()


def test_packager_exact_allowlist_never_includes_weights_or_data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(package, 'validate_readiness', lambda *args: None)
    monkeypatch.setattr(package, 'validate_comparison', lambda *args: None)
    payloads = {}
    for name in REQUIRED_MEMBERS:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        payloads[name] = ('# '+name+'\n').encode()
        path.write_bytes(payloads[name])
    (tmp_path/'private.pt').write_bytes(b'weights')
    (tmp_path/'raw-video.mp4').write_bytes(b'private media')
    monkeypatch.setattr(package, 'implementation_hash', lambda: implementation(payloads))
    result = package.run(tmp_path/'export')
    with zipfile.ZipFile(tmp_path/'export'/result['overlay_name']) as archive:
        assert set(archive.namelist()) == set(REQUIRED_MEMBERS) | {'overlay-sha256.json'}
        for name, digest in result['member_hashes'].items():
            assert sha(archive.read(name)) == digest
    assert not result['contains_dataset'] and not result['contains_existing_weights']
    with pytest.raises(FileExistsError):
        package.run(tmp_path/'export')
