"""Failure boundaries must be exact even after partially mutated optimizer state."""
import json
import random

import numpy as np
import pytest
import torch

from genie_diep.baseline import Trainer
from test_baseline_pipeline import project, write


def assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    elif isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for key in expected:assert_equal(actual[key],expected[key])
    elif isinstance(expected,(list,tuple)):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):assert_equal(a,b)
    else:assert actual==expected


@pytest.mark.parametrize('failure',['loss','gradient','backward','optimizer','scheduler'])
def test_failed_update_restores_every_state_and_replays_after_save(project,monkeypatch,failure):
    trainer=Trainer(project,'spec.json','tokenizer',preflight=True)
    trainer.update();trainer.save(project/'before.pt')
    expected_loss=trainer.update()
    expected=trainer._update_boundary()
    failed=Trainer(project,'spec.json','tokenizer',preflight=True,resume=project/'before.pt')
    before=failed._update_boundary();calls=0
    if failure in {'loss','backward'}:
        original=failed.loss
        def faulty_loss(x):
            nonlocal calls
            calls+=1
            value=original(x)
            if calls==2:
                random.random();np.random.random();torch.rand(3)
                if failure=='loss':return value*float('nan')
                def reject(gradient):raise RuntimeError('injected backward failure')
                value.register_hook(reject)
            return value
        monkeypatch.setattr(failed,'loss',faulty_loss)
    elif failure=='gradient':
        handle=failed.parameters[0].register_hook(lambda gradient:gradient*float('inf'))
    elif failure=='optimizer':
        original=failed.optimizer.step
        def partial_step(*args,**kwargs):
            original(*args,**kwargs)
            raise RuntimeError('injected partial step failure')
        monkeypatch.setattr(failed.optimizer,'step',partial_step)
    else:
        original=type(failed.scheduler).step
        def partial_step(self,*args,**kwargs):
            original(self,*args,**kwargs)
            raise RuntimeError('injected partial step failure')
        monkeypatch.setattr(type(failed.scheduler),'step',partial_step)
    with pytest.raises((FloatingPointError,RuntimeError)):
        failed.update()
    assert_equal(failed._update_boundary(),before)
    assert all(p.grad is None for p in failed.parameters)
    diagnostic=failed.last_failure
    assert diagnostic['restored_pre_update_boundary'] is True
    assert diagnostic['attempted_step']==2
    assert len(diagnostic['sampled_windows'])==failed.microbatch*failed.accumulation
    if failure=='gradient':
        assert diagnostic['affected_gradients'][0]['nonfinite']>0
        assert diagnostic['affected_gradients'][0]['name'].startswith('tokenizer.')
        handle.remove()
    failed.save(project/'failed.pt')
    json.dumps(diagnostic,allow_nan=False)
    monkeypatch.undo()
    assert failed.update()==expected_loss
    assert failed.last_failure is None
    assert_equal(failed._update_boundary(),expected)
    resumed=Trainer(project,'spec.json','tokenizer',preflight=True,resume=project/'failed.pt')
    assert resumed.last_failure==diagnostic
    assert resumed.update()==expected_loss
    assert_equal(resumed._update_boundary(),expected)


def test_budget_stop_restores_partial_accumulation_and_rng(project):
    trainer=Trainer(project,'spec.json','dynamics',preflight=True)
    before=trainer._update_boundary()
    class Budget:
        calls=0
        def permits(self):
            self.calls+=1
            return self.calls==1
    assert trainer.update(Budget()) is None
    assert_equal(trainer._update_boundary(),before)
    assert trainer.last_failure is None
    trainer.save(project/'budget.pt')
    expected_loss=trainer.update();expected=trainer._update_boundary()
    resumed=Trainer(project,'spec.json','dynamics',preflight=True,resume=project/'budget.pt')
    assert resumed.update()==expected_loss
    assert_equal(resumed._update_boundary(),expected)


def test_enabled_scaler_can_unscale_again_after_failed_update(project,monkeypatch):
    # CPU GradScaler exercises the same per-optimizer state machine without CUDA.
    monkeypatch.setattr(Trainer,'_new_scaler',lambda self:torch.amp.GradScaler('cpu',init_scale=self.scaler_initial_scale))
    trainer=Trainer(project,'spec.json','tokenizer',preflight=True)
    trainer.update();before=trainer._update_boundary();old_scaler=trainer.scaler
    handle=trainer.parameters[0].register_hook(lambda gradient:gradient*float('inf'))
    with pytest.raises(FloatingPointError,match='Nonfinite gradient'):trainer.update()
    handle.remove()
    assert trainer.scaler is not old_scaler
    assert_equal(trainer._update_boundary(),before)
    # Reusing the failed scaler would raise "unscale_() has already been called".
    assert np.isfinite(trainer.update())
    assert trainer.step==before['step']+1


def test_enabled_scaler_growth_stays_bounded_across_updates(project,monkeypatch):
    spec=json.loads((project/'spec.json').read_text())
    spec['optimization'].update(fp16_initial_scale=512,fp16_max_scale=1024);write(project/'spec.json',spec)
    monkeypatch.setattr(Trainer,'_new_scaler',lambda self:torch.amp.GradScaler('cpu',init_scale=self.scaler_initial_scale,growth_interval=1))
    trainer=Trainer(project,'spec.json','tokenizer',preflight=True)
    assert trainer.scaler.get_scale()==512
    for step in range(3):
        assert np.isfinite(trainer.update())
        assert trainer.step==step+1
        assert trainer.scaler.get_scale()==1024


@pytest.mark.parametrize('maximum',[0,512,float('nan'),float('inf')])
def test_invalid_maximum_scale_rejected(project,maximum):
    spec=json.loads((project/'spec.json').read_text())
    spec['optimization']['fp16_max_scale']=maximum;write(project/'spec.json',spec)
    with pytest.raises(ValueError,match='maximum scale'):
        Trainer(project,'spec.json','tokenizer',preflight=True)


def test_failed_rollback_refuses_to_overwrite_last_checkpoint(project,monkeypatch):
    trainer=Trainer(project,'spec.json','tokenizer',preflight=True)
    path=project/'last.pt';trainer.save(path);original=path.read_bytes()
    monkeypatch.setattr(trainer,'loss',lambda x:torch.tensor(float('nan')))
    def broken_restore(*args):raise RuntimeError('injected restore failure')
    monkeypatch.setattr(trainer,'_restore_update_boundary',broken_restore)
    with pytest.raises(RuntimeError,match='restore failure'):trainer.update()
    with pytest.raises(RuntimeError,match='Cannot save an incomplete'):trainer.save(path)
    assert path.read_bytes()==original


@pytest.mark.parametrize('scale',[0,-1,float('nan'),float('inf')])
def test_invalid_initial_scale_rejected(project,scale):
    spec=json.loads((project/'spec.json').read_text())
    spec['optimization']['fp16_initial_scale']=scale;write(project/'spec.json',spec)
    with pytest.raises(ValueError,match='finite and positive'):
        Trainer(project,'spec.json','tokenizer',preflight=True)
