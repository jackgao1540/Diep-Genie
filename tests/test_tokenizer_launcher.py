"""Budget and production-command checks without allocating cloud compute."""
import json
import pytest

from scripts.build_tokenizer_notebook import build, notebook
import test_drive_trial_launcher as trial


PILOT = dict(status='passed', implementation_sha256='implementation',
             spec_sha256='spec', binding_sha256='binding', private_unused='omit')


@pytest.mark.parametrize('seconds,units,rate,balance', [
    (21500, 0, 1, 100), (0, 7.99, 1, 100), (0, 0, 200, 100), (0, 0, 1, 50.02)])
def test_all_budget_limits_release_before_mount(monkeypatch, tmp_path, seconds, units, rate, balance):
    monkeypatch.setattr(trial, 'notebook', lambda: notebook(PILOT))
    colab, run, scope, code = trial.launcher_environment(monkeypatch, tmp_path)
    scope.update(PRIOR_ALLOCATED_SECONDS=seconds, PRIOR_COMPUTE_UNITS=units,
                 DISPLAYED_UNITS_PER_HOUR=rate, LIVE_BALANCE=balance)
    with pytest.raises(AssertionError, match='Insufficient budget'):
        exec(compile(code, 'launcher', 'exec'), scope)
    colab.drive.mount.assert_not_called()
    run.assert_not_called()
    colab.runtime.unassign.assert_called_once()
    remaining = scope['deadline']-scope['allocation_start']
    assert remaining <= 21600-seconds+1e-6
    assert remaining*rate/3600 <= 8-units+1e-6
    assert remaining*rate/3600 <= balance-50+1e-6


def test_corrupt_archive_releases_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(trial, 'notebook', lambda: notebook(PILOT))
    trial.test_incomplete_upload_stops_before_install_and_releases_runtime(monkeypatch, tmp_path)


def test_fresh_production_stage_and_disabled_export(tmp_path):
    pilot = tmp_path/'pilot.json'
    pilot.write_text(json.dumps(PILOT))
    output = tmp_path/'export'
    build(output, pilot)
    document = json.loads((output/'tokenizer-v9-bounded-v1.ipynb').read_text())
    scope = {}
    exec(''.join(document['cells'][1]['source']), scope)
    assert scope['AUTHORIZED_TO_RUN'] is False
    code = ''.join(document['cells'][2]['source'])
    assert "'--stage', 'tokenizer'" in code and "'--updates', '20000'" in code
    assert "'--authorized'" in code and "'--precision', 'fp16'" in code
    assert '--preflight' not in code and '--resume' not in code and '--initial' not in code
    assert 'private_unused' not in code
    assert 'runtime.unassign()' in code and 'process.terminate()' in code
    with pytest.raises(FileExistsError):
        build(output, pilot)


def test_reject_failed_pilot():
    with pytest.raises(ValueError, match='Passing GPU'):
        notebook(dict(PILOT, status='failed'))
