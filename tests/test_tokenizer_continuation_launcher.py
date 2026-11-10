"""Local-only checks for private continuation packaging and shared-budget cleanup."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest
import torch

from scripts.build_drive_trial_notebook import ARCHIVE_BYTES, ARCHIVE_SHA256
from scripts.build_tokenizer_continuation_notebook import BOUND_FILES, HASH_FIELDS, PARENT_MEMBERS, REQUIRED_MEMBERS, build, notebook
import scripts.package_tokenizer_continuation as package
import test_drive_trial_launcher as trial


def sha(content):
    return hashlib.sha256(content).hexdigest()


def implementation(payloads):
    digest = hashlib.sha256()
    for name in ['baseline.py', 'cohort.py', 'model.py', 'baseline_evaluation.py']:
        digest.update(name.encode())
        digest.update(payloads['genie_diep/'+name].replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def fixture_manifest():
    return dict(overlay_name='genie-tokenizer-continuation-v1.zip', overlay_sha256='a'*64,
                overlay_bytes=100, **{field: 'b'*64 for field in HASH_FIELDS})


def environment(monkeypatch, tmp_path, extra=None, missing=None, wrong_hash=False, wrong_parent_binding=False):
    drive = tmp_path/'content/drive/MyDrive'
    drive.mkdir(parents=True)
    base = drive/'genie-baseline-v1-v9.zip'
    with zipfile.ZipFile(base, 'w') as archive:
        archive.writestr('data/frozen.npz', b'frozen dataset')
        archive.writestr('genie_diep/model.py', b'original source')
        archive.writestr('bundle-sha256.json', json.dumps({
            'data/frozen.npz': sha(b'frozen dataset'), 'genie_diep/model.py': sha(b'original source')}))
    payloads = {name: ('# '+name+'\n').encode() for name in REQUIRED_MEMBERS}
    payloads['genie_diep/model.py'] = b'# revised source\r\n'
    if extra:
        payloads.update(extra)
    if missing:
        payloads.pop(missing)
    hashes = {name: sha(payload) for name, payload in payloads.items()}
    manifest = fixture_manifest()
    manifest.update({field: hashes.get(name, 'f'*64) for field, name in BOUND_FILES.items()})
    manifest['expected_implementation_sha256'] = implementation(payloads)
    if wrong_hash:
        hashes[PARENT_MEMBERS['detail']] = '0'*64
    if wrong_parent_binding:
        manifest['expected_baseline_parent_sha256'] = '0'*64
    overlay = drive/manifest['overlay_name']
    with zipfile.ZipFile(overlay, 'w') as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
        archive.writestr('overlay-sha256.json', json.dumps(hashes))
    manifest.update(overlay_sha256=sha(overlay.read_bytes()), overlay_bytes=overlay.stat().st_size)
    monkeypatch.setattr(trial, 'notebook', lambda: notebook(manifest))
    colab, run, scope, code = trial.launcher_environment(monkeypatch, tmp_path)
    code = code.replace(ARCHIVE_SHA256, sha(base.read_bytes())).replace(str(ARCHIVE_BYTES), str(base.stat().st_size))
    return colab, run, scope, code, payloads


@pytest.mark.parametrize('outcome', ['completed', 'budget_exhausted'])
def test_success_or_bounded_stop_binds_parents_and_releases(monkeypatch, tmp_path, outcome):
    colab, run, scope, code, payloads = environment(monkeypatch, tmp_path)
    scope.update(PRIOR_ALLOCATED_SECONDS=100, PRIOR_COMPUTE_UNITS=.1)
    def execute(command, **kwargs):
        if 'pip' in command:
            return
        assert 'scripts.run_tokenizer_continuation' in command and '--authorized' in command
        assert command[command.index('--experiment')+1] == 'configs/tokenizer-continuation-v1.json'
        assert command[command.index('--device')+1] == 'cuda'
        assert float(command[command.index('--allocation-start')+1]) == scope['accounting_start']
        assert scope['accounting_start'] <= scope['allocation_start'] - 100
        assert scope['deadline'] <= scope['accounting_start'] + 10800
        assert (scope['deadline'] - scope['accounting_start']) * scope['DISPLAYED_UNITS_PER_HOUR']/3600 <= 4
        output = Path(command[command.index('--out')+1]); output.mkdir(parents=True)
        (output/'continuation-report.json').write_text(json.dumps(dict(status=outcome)))
    run.side_effect = execute
    exec(compile(code, 'launcher', 'exec'), scope)
    colab.runtime.unassign.assert_called_once()
    root = tmp_path/'content/genie-tokenizer-continuation-v1'
    assert (root/'data/frozen.npz').read_bytes() == b'frozen dataset'
    for name in PARENT_MEMBERS.values():
        assert (root/name).read_bytes() == payloads[name]
    status = json.loads(next(tmp_path.rglob('launch-status.json')).read_text())
    assert status['status'] == outcome and status['overlay_applied']
    assert status['maximum_compute_units'] == 4 and status['maximum_allocated_seconds'] == 10800
    assert status['maximum_additional_optimizer_calls_per_arm'] == 4002 and status['target_step'] == 5000
    assert status['quality_gates'] == 'unmeasured' and status['training_stage_promotion'] is False


@pytest.mark.parametrize('extra,missing,wrong_hash,wrong_parent,message', [
    ({'data/frozen.npz': b'changed'}, None, False, False, 'exactly'),
    ({'continuation-inputs/extra.pt': b'other weights'}, None, False, False, 'exactly'),
    ({'../escape.py': b'escape'}, None, False, False, 'exactly'),
    (None, PARENT_MEMBERS['detail'], False, False, 'exactly'),
    (None, None, True, False, 'member hash mismatch'),
    (None, None, False, True, 'Input binding mismatch'),
])
def test_invalid_overlay_never_writes_sources_or_parents(monkeypatch, tmp_path, extra, missing, wrong_hash, wrong_parent, message):
    colab, run, scope, code, _ = environment(monkeypatch, tmp_path, extra, missing, wrong_hash, wrong_parent)
    with pytest.raises(AssertionError, match=message):
        exec(compile(code, 'launcher', 'exec'), scope)
    colab.runtime.unassign.assert_called_once(); run.assert_not_called()
    root = tmp_path/'content/genie-tokenizer-continuation-v1'
    assert (root/'genie_diep/model.py').read_bytes() == b'original source'
    assert (root/'data/frozen.npz').read_bytes() == b'frozen dataset'
    assert all(not (root/name).exists() for name in PARENT_MEMBERS.values())


def test_timeout_saves_failure_and_releases(monkeypatch, tmp_path):
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


@pytest.mark.parametrize('seconds,units,rate,balance', [(10700,0,1,100), (0,3.99,1,100), (0,0,100,100), (0,0,1,50.01)])
def test_prior_debits_and_reserve_release_before_drive(monkeypatch, tmp_path, seconds, units, rate, balance):
    monkeypatch.setattr(trial, 'notebook', lambda: notebook(fixture_manifest()))
    colab, run, scope, code = trial.launcher_environment(monkeypatch, tmp_path)
    scope.update(PRIOR_ALLOCATED_SECONDS=seconds, PRIOR_COMPUTE_UNITS=units,
                 DISPLAYED_UNITS_PER_HOUR=rate, LIVE_BALANCE=balance)
    with pytest.raises(AssertionError, match='Insufficient budget'):
        exec(compile(code, 'launcher', 'exec'), scope)
    colab.runtime.unassign.assert_called_once(); colab.drive.mount.assert_not_called(); run.assert_not_called()


def test_invalid_accounting_also_releases(monkeypatch, tmp_path):
    monkeypatch.setattr(trial, 'notebook', lambda: notebook(fixture_manifest()))
    colab, run, scope, code = trial.launcher_environment(monkeypatch, tmp_path)
    scope['LIVE_BALANCE'] = float('nan')
    with pytest.raises(AssertionError):
        exec(compile(code, 'launcher', 'exec'), scope)
    colab.runtime.unassign.assert_called_once(); colab.drive.mount.assert_not_called(); run.assert_not_called()


def test_export_is_disabled_private_and_cannot_overwrite(tmp_path):
    manifest = tmp_path/'overlay.json'; manifest.write_text(json.dumps(fixture_manifest()))
    result = build(tmp_path/'launcher', manifest)
    document = json.loads((tmp_path/'launcher'/result['notebook']).read_text())
    scope = {}; exec(''.join(document['cells'][1]['source']), scope)
    source = ''.join(document['cells'][2]['source'])
    assert scope['AUTHORIZED_TO_RUN'] is False
    assert source.count('threading.Thread(target=watchdog') == 1
    assert source.index('threading.Thread(target=watchdog') < source.index("drive.mount('/content/drive')")
    assert 'stop_at-time.time()-180' in source
    assert result['maximum_compute_units'] == 4 and result['maximum_allocated_seconds'] == 10800
    assert result['contains_private_checkpoint_weights'] and not result['training_stage_promotion']
    with pytest.raises(FileExistsError):
        build(tmp_path/'launcher', manifest)


@pytest.mark.parametrize('name', ['../overlay.zip', 'genie-baseline-v1-v9.zip', 'genie-tokenizer-comparison-v1.zip'])
def test_filename_cannot_replace_previous_artifacts(name):
    with pytest.raises(ValueError, match='filename'):
        notebook(dict(fixture_manifest(), overlay_name=name))


def test_packager_binds_only_two_parents_and_preserves_originals(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path); payloads = {}
    for name in REQUIRED_MEMBERS - set(PARENT_MEMBERS.values()):
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True)
        payloads[name] = ('# '+name+'\n').encode(); path.write_bytes(payloads[name])
    checkpoints = {}
    plan = {'parents': {}}
    for arm, name in PARENT_MEMBERS.items():
        path = tmp_path/f'preserved-{arm}.pt'; path.write_bytes(('preserved '+arm).encode())
        checkpoints[arm] = path; payloads[name] = path.read_bytes()
        plan['parents'][arm] = dict(path=name, sha256=sha(path.read_bytes()))
    monkeypatch.setattr(package, 'validate_inputs', lambda *args: (plan, checkpoints))
    monkeypatch.setattr(package, 'implementation_hash', lambda: implementation(payloads))
    (tmp_path/'unrelated-private.pt').write_bytes(b'never include')
    result = package.run(tmp_path/'export', checkpoints['baseline'], checkpoints['detail'])
    with zipfile.ZipFile(tmp_path/'export'/result['overlay_name']) as archive:
        assert set(archive.namelist()) == set(REQUIRED_MEMBERS) | {'overlay-sha256.json'}
        for name, digest in result['member_hashes'].items():
            assert sha(archive.read(name)) == digest
    assert result['parent_checkpoints_verified'] and result['contains_private_checkpoint_weights']
    for arm, path in checkpoints.items():
        assert sha(path.read_bytes()) == plan['parents'][arm]['sha256']
    with pytest.raises(FileExistsError):
        package.run(tmp_path/'export', checkpoints['baseline'], checkpoints['detail'])


@pytest.mark.parametrize('fault,message', [('hash','hash changed'), ('step','step-999'), ('preflight','step-999'), ('nonfinite','Nonfinite'), ('sampler',None)])
def test_local_parent_checks_reject_wrong_inputs(monkeypatch, tmp_path, fault, message):
    parent_path = tmp_path/'parent-experiment.json'; parent_path.write_text('{}')
    plan = dict(parent_experiment=dict(path=parent_path.name, sha256=sha(parent_path.read_bytes())), parents={})
    paths = {}
    for arm in ('baseline', 'detail'):
        state = dict(step=999, contract=dict(stage='tokenizer', preflight=False),
                     model={'weight': torch.zeros(1)}, optimizer={}, sampler={'draws': 15985})
        if arm == 'detail':
            if fault == 'step': state['step'] = 998
            if fault == 'preflight': state['contract']['preflight'] = True
            if fault == 'nonfinite': state['model']['weight'].fill_(float('inf'))
            if fault == 'sampler': state['sampler']['draws'] += 1
        path = tmp_path/f'{arm}.pt'; torch.save(state, path); paths[arm] = path
        plan['parents'][arm] = dict(path=PARENT_MEMBERS[arm], sha256=sha(path.read_bytes()))
    if fault == 'hash': plan['parents']['detail']['sha256'] = '0'*64
    monkeypatch.setitem(sys.modules, 'scripts.run_tokenizer_continuation', SimpleNamespace(
        validate_experiment=lambda *args: plan, validate_parent=lambda *args: None,
        validate_continuation_readiness=lambda *args: None))
    monkeypatch.setattr(package, 'validate_readiness', lambda *args: None)
    with pytest.raises((ValueError, AssertionError), match=message):
        package.validate_inputs(tmp_path, paths['baseline'], paths['detail'])
