import json
from types import SimpleNamespace

import pytest
import torch

from scripts.run_tokenizer_continuation import (
    ContinuationTrainer, allocation_budget, inside, validate_experiment,
    validate_continuation_readiness,
)


def test_continuation_budget_counts_setup_and_shared_rate(monkeypatch):
    monkeypatch.setattr('scripts.run_tokenizer_continuation.time.time', lambda:700.)
    monkeypatch.setattr('genie_diep.baseline.time.monotonic', lambda:10.)
    args = SimpleNamespace(preflight=False, authorized=True, device='cuda', rate=2., balance=100., allocation_start=100.)
    # Four CU permits 7200 allocated seconds, of which setup spent 600; keep export reserve.
    assert allocation_budget(args).remaining() == pytest.approx(6420.)
    args.authorized = False
    with pytest.raises(ValueError, match='Authorized'):
        allocation_budget(args)


@pytest.mark.parametrize('field,value', [('rate',0.), ('rate',float('nan')), ('balance',50.),
                                       ('allocation_start',None), ('allocation_start',1000000.)])
def test_bad_accounting_cannot_launch(field, value, monkeypatch):
    monkeypatch.setattr('scripts.run_tokenizer_continuation.time.time', lambda:700.)
    args = SimpleNamespace(preflight=False, authorized=True, device='cuda', rate=2., balance=100., allocation_start=100.)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        allocation_budget(args)


def test_preflight_cannot_use_gpu():
    with pytest.raises(ValueError, match='CPU-only'):
        allocation_budget(SimpleNamespace(preflight=True, device='cuda'))


def test_experiment_cannot_expand_target(tmp_path):
    (tmp_path/'plan.json').write_text(json.dumps({'parent_step':999, 'target_step':5001}))
    with pytest.raises(ValueError, match='boundaries'):
        validate_experiment(tmp_path, 'plan.json')
    with pytest.raises(ValueError, match='escapes'):
        inside(tmp_path, '../outside.pt')


def test_production_requires_bound_local_execution_evidence(tmp_path):
    from genie_diep.data import file_hash
    (tmp_path/'configs').mkdir()
    (tmp_path/'plan.json').write_text('{}')
    evidence = dict(status='passed', experiment_sha256=file_hash(tmp_path/'plan.json'),
                    parents_unchanged=True, full_suite_passed=True, exact_resume_both_arms=True,
                    maximum_state_difference=0)
    path = tmp_path/'configs/tokenizer-continuation-readiness-v1.json'
    path.write_text(json.dumps(evidence))
    validate_continuation_readiness(tmp_path, 'plan.json')
    evidence['maximum_state_difference'] = 1e-8
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match='exact-resume'):
        validate_continuation_readiness(tmp_path, 'plan.json')
    evidence['maximum_state_difference'] = 0
    path.write_text(json.dumps(evidence))
    (tmp_path/'plan.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='stale'):
        validate_continuation_readiness(tmp_path, 'plan.json')


def test_same_metrics_assess_both_splits_and_restore_training_mode():
    class Model(torch.nn.Module):
        def tokenizer(self, x):
            return x*.5, None, None
    x = torch.zeros(16,3,4,4)
    x[:,0] = .8
    x[8:,1] = .4
    cohort = SimpleNamespace(locate=lambda descriptor:(0,0), window=lambda *args:x)
    model = Model()
    trainer = SimpleNamespace(model=model, module=model, device=torch.device('cpu'), step=999,
                              data=cohort, validation=cohort,
                              evaluation={'training_calibration':[{'source_id':'a'}],
                                          'validation':[{'source_id':'v'}]})
    train = ContinuationTrainer.assess(trainer, 'train')
    val = ContinuationTrainer.assess(trainer, 'val')
    assert train['complete'] and val['complete'] and model.training
    assert train['mse'] == pytest.approx(float((x*.5).square().mean()))
    assert train['foreground_mse'] == pytest.approx(train['mse'])
    assert train['temporal_mse'] == pytest.approx(float((torch.diff(x,dim=0)*.5).square().mean()))
    assert (train['mse'],train['foreground_mse'],train['temporal_mse']) == (val['mse'],val['foreground_mse'],val['temporal_mse'])
    with pytest.raises(ValueError, match='reserved test'):
        ContinuationTrainer.assess(trainer, 'test')


def test_assessment_reports_budget_interruption_without_partial_pass():
    class Model(torch.nn.Module):
        def tokenizer(self,x):return x,None,None
    model = Model()
    trainer = SimpleNamespace(model=model,module=model,device=torch.device('cpu'),step=999,
                              data=None,validation=None,evaluation={'validation':[{'source_id':'v'}]})
    result = ContinuationTrainer.assess(trainer,'val',SimpleNamespace(permits=lambda *args:False))
    assert result['complete'] is False and result['samples']==0 and result['mse'] is None
    assert model.training
