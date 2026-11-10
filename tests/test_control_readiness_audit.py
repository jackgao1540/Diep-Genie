import copy
import json

import pytest
import torch

from scripts.audit_control_readiness import (
    compare_modules, control_findings, fresh_tokenizer_provenance, run, sha256,
    validate_checkpoint)


def weights():
    return {name + '.weight': torch.tensor([1.0, 2.0])
            for name in ('tokenizer', 'lam', 'dynamics')}


def checkpoint():
    return dict(config={'width': 128, 'action_codes': [9, 8, 2]}, step=518,
                contract=dict(stage='tokenizer', preflight=True,
                              binding_sha256='binding', evaluation_sha256='selection'),
                model=weights())


def test_module_comparison_measures_frozen_weights_and_tokenizer_change():
    reference = weights()
    current = weights()
    current['tokenizer.weight'][0] += 0.25
    report = compare_modules(current, reference)
    assert report['tokenizer']['maximum_absolute_difference'] == 0.25
    assert report['tokenizer']['exactly_unchanged'] is False
    assert report['lam'] == dict(tensors=1, elements=2, exactly_unchanged=True,
                                 maximum_absolute_difference=0.0)


@pytest.mark.parametrize('mutation', ['dtype', 'shape', 'missing', 'nonfinite'])
def test_incompatible_or_invalid_tensors_are_rejected(mutation):
    reference, current = weights(), weights()
    if mutation == 'dtype':
        current['lam.weight'] = current['lam.weight'].double()
    elif mutation == 'shape':
        current['lam.weight'] = current['lam.weight'][:1]
    elif mutation == 'missing':
        del current['lam.weight']
    else:
        current['lam.weight'][0] = float('nan')
    with pytest.raises(ValueError):
        compare_modules(current, reference)


@pytest.mark.parametrize('mutation', ['architecture', 'stage', 'binding', 'selection', 'preflight'])
def test_checkpoint_contract_mismatch_is_rejected(mutation):
    state = checkpoint()
    config = copy.deepcopy(state['config'])
    if mutation == 'architecture':
        state['config']['width'] = 64
    else:
        key = {'stage': 'stage', 'binding': 'binding_sha256',
               'selection': 'evaluation_sha256', 'preflight': 'preflight'}[mutation]
        state['contract'][key] = 'wrong'
    with pytest.raises(ValueError):
        validate_checkpoint(state, config, 'binding', 'selection')


def test_unchanged_weights_cannot_pass_control_or_establish_initialization_alone():
    modules = compare_modules(weights(), weights())
    report = control_findings(modules, False)
    assert report['status'] == 'not_eligible_for_playable_control'
    assert report['action_and_dynamics_untrained'] is False
    assert report['G2']['status'] == report['G3']['status'] == 'unmeasured'
    assert report['commands_evaluated'] == report['rollout_frames_evaluated'] == 0
    assert report['G2']['eligible'] is report['G3']['eligible'] is False


def test_recorded_fresh_tokenizer_run_plus_unchanged_modules_proves_untrained_status():
    state = checkpoint()
    record = dict(contract=state['contract'],
                  arguments=dict(stage='tokenizer', initial=None, resume=None))
    assert fresh_tokenizer_provenance(record, state)
    report = control_findings(compare_modules(weights(), weights()), True)
    assert report['action_and_dynamics_untrained'] is True
    assert report['G2']['eligible'] is False
    record['arguments']['initial'] = 'previous.pt'
    assert not fresh_tokenizer_provenance(record, state)


def test_changed_frozen_modules_are_flagged_without_control_claims():
    current = weights()
    current['dynamics.weight'][0] += 1
    report = control_findings(compare_modules(current, weights()), True)
    assert report['frozen_modules_consistent'] is False
    assert report['action_and_dynamics_untrained'] is False
    assert report['G3']['eligible'] is False


def test_audit_writes_bound_read_only_evidence_and_refuses_overwrite(tmp_path):
    configs = tmp_path / 'configs'
    configs.mkdir()
    binding_path = configs / 'binding.json'
    binding_path.write_text('{}', encoding='utf-8')
    selection_path = configs / 'selection.json'
    selection_path.write_text(json.dumps(dict(binding_sha256=sha256(binding_path),
        validation=[dict(source_id='held-out')],
        training_calibration=[dict(source_id='training')])), encoding='utf-8')
    current, prior = checkpoint(), checkpoint()
    prior['step'] = 219
    prior['contract']['preflight'] = False
    for state in (current, prior):
        state['contract']['binding_sha256'] = sha256(binding_path)
        state['contract']['evaluation_sha256'] = sha256(selection_path)
    (configs / 'training-baseline-v1.json').write_text(json.dumps(dict(
        model=current['config'], data_binding='configs/binding.json',
        evaluation_selection='configs/selection.json')), encoding='utf-8')
    checkpoint_path, reference_path = tmp_path / 'current.pt', tmp_path / 'prior.pt'
    torch.save(current, checkpoint_path)
    torch.save(prior, reference_path)
    (tmp_path / 'run.json').write_text(json.dumps(dict(contract=prior['contract'],
        arguments=dict(stage='tokenizer', initial=None, resume=None))), encoding='utf-8')
    before = (sha256(checkpoint_path), sha256(reference_path))
    output = tmp_path / 'output' / 'audit.json'
    report = run(tmp_path, checkpoint_path, reference_path, output)
    assert report['control']['action_and_dynamics_untrained'] is True
    assert report['optimizer_updates'] == 0
    assert report['checkpoint']['sha256'] == before[0]
    assert report['reference']['sha256'] == before[1]
    assert before == (sha256(checkpoint_path), sha256(reference_path))
    assert report['calibration_sources'] == report['validation_sources'] == 1
    assert json.loads(output.read_text(encoding='utf-8')) == report
    with pytest.raises(FileExistsError):
        run(tmp_path, checkpoint_path, reference_path, output)
