import copy
from pathlib import Path

import pytest
import torch

from genie_diep.baseline import Budget
from scripts.recover_staged_tokenizer_fp32 import (
    ExceptionTrainer, POLICY, KEYS, convert_state, load_boundary, reference_update,
    exact_replay, finite_tree, cpu_copy,
    anchor_inactive,
)
from scripts.run_staged_tokenizer import StagedTrainer
from scripts.run_tokenizer_comparison import compare_exact
from test_staged_tokenizer_runner import staged
from test_baseline_pipeline import project


def context(root):
    return dict(manifest_sha256='manifest',failed_proof_sha256='failed-proof',out=str(root/'events'))


def budget():return Budget(dict(max_allocated_hours=1,max_compute_units=1),reserve_seconds=0)


def test_fp32_exception_equals_independent_reference_and_preserves_amp(staged):
    original=StagedTrainer(staged,'plan.json','joint',local_check=True,mode='full')
    original.update();original.save(staged/'original.pt')
    saved=torch.load(staged/'original.pt',weights_only=True)
    reference=reference_update(original,budget())
    actual=ExceptionTrainer(staged,'plan.json','joint',local_check=True,mode='full',context=context(staged))
    converted=convert_state(saved,actual.contract);load_boundary(actual,converted)
    loss=actual.fp32_update(budget(),dict(kind='local_test'))
    assert loss==reference['loss'] and actual.precision=='fp32'
    boundary=cpu_copy(actual._update_boundary())
    for key in KEYS:compare_exact(boundary[key],reference['boundary'][key])
    assert len(actual.fp32_events)==1 and actual.fp32_events[0]['dropped_samples']==0
    compare_exact(actual.scaler.state_dict(),saved['scaler'])
    actual.save(staged/'exception.pt')
    resumed=ExceptionTrainer(staged,'plan.json','joint',local_check=True,mode='full',context=context(staged),resume=staged/'exception.pt')
    compare_exact(resumed.fp32_events,actual.fp32_events)
    compare_exact(resumed._update_boundary(),actual._update_boundary())


def test_conversion_changes_contract_and_event_metadata_only(staged):
    original=StagedTrainer(staged,'plan.json','joint',local_check=True);original.save(staged/'original.pt')
    saved=torch.load(staged/'original.pt',weights_only=True);saved['scaler']=dict(scale=1024.,_growth_tracker=1803)
    before=copy.deepcopy(saved);converted=convert_state(saved,{'different':'provenance'})
    for key in saved:
        if key!='contract':compare_exact(saved[key],converted[key])
    assert converted['fp32_exception_events']==[]
    compare_exact(saved,before)


def test_original_amp1024_tracker_is_preserved_across_exception(staged,monkeypatch):
    class StoredScaler:
        def __init__(self):self.state=dict(scale=1024.,growth_factor=2.,backoff_factor=.5,growth_interval=2000,_growth_tracker=1803)
        def state_dict(self):return copy.deepcopy(self.state)
        def load_state_dict(self,state):self.state=copy.deepcopy(state)
    trainer=ExceptionTrainer(staged,'plan.json','joint',local_check=True,context=context(staged))
    trainer.precision='fp16';trainer.scaler=StoredScaler();saved=trainer.scaler.state_dict()
    monkeypatch.setattr(trainer,'_new_scaler',StoredScaler)
    trainer.fp32_update(budget(),dict(kind='stored_AMP_test'))
    assert trainer.precision=='fp16' and trainer.scaler.state_dict()==saved
    assert trainer.step==1 and trainer.sampler.draws==17


def test_event_replay_restores_event_history(staged):
    trainer=ExceptionTrainer(staged,'plan.json','joint',local_check=True,context=context(staged))
    # Force the real initial exception dispatch at the fixture's first update.
    from scripts import recover_staged_tokenizer_fp32 as recovery
    original=POLICY['initial_exception_update'];POLICY['initial_exception_update']=1
    folder=staged/'replay';folder.mkdir()
    try:
        result=exact_replay(trainer,folder,budget())
        assert result['passed'] and trainer.step==1 and len(trainer.fp32_events)==1
    finally:POLICY['initial_exception_update']=original


def test_event_cap_refuses_before_sampling(staged):
    trainer=ExceptionTrainer(staged,'plan.json','joint',local_check=True,context=context(staged))
    trainer.fp32_events=[{} for _ in range(4)];before=trainer._update_boundary()
    with pytest.raises(ValueError,match='cap reached'):trainer.update(budget())
    compare_exact(before,trainer._update_boundary())


def test_recurrent_overflow_requires_exact_rollback_and_uses_fp32(staged,monkeypatch):
    trainer=ExceptionTrainer(staged,'plan.json','joint',local_check=True,context=context(staged));trainer.precision='fp16'
    method=StagedTrainer.update
    def injected(self,allocation):
        if self.precision=='fp32':return method(self,allocation)
        self.last_failure=dict(phase='gradient_check',message='Nonfinite gradient; no optimizer update',restored_pre_update_boundary=True)
        raise FloatingPointError('injected overflow')
    monkeypatch.setattr(StagedTrainer,'update',injected)
    assert trainer.update(budget()) is not None
    assert trainer.precision=='fp16' and trainer.step==1 and len(trainer.fp32_events)==1
    assert trainer.fp32_events[0]['reason']['exact_rollback'] is True


def test_forward_failure_cannot_trigger_exception(staged,monkeypatch):
    trainer=ExceptionTrainer(staged,'plan.json','joint',local_check=True,context=context(staged));trainer.precision='fp16'
    def injected(self,allocation):
        self.last_failure=dict(phase='forward',message='Nonfinite loss; no optimizer update',restored_pre_update_boundary=True)
        raise FloatingPointError('injected forward failure')
    monkeypatch.setattr(StagedTrainer,'update',injected)
    with pytest.raises(ValueError,match='Not an eligible'):trainer.update(budget())
    assert not trainer.fp32_events and trainer.step==0


def test_fp32_failure_rolls_back_full_state_and_preserves_original_precision(staged,monkeypatch):
    trainer=ExceptionTrainer(staged,'plan.json','joint',local_check=True,context=context(staged));before=trainer._update_boundary()
    def broken_loss(x):return next(trainer.model.parameters()).sum()*float('nan')
    monkeypatch.setattr(trainer,'loss',broken_loss)
    with pytest.raises(FloatingPointError):trainer.fp32_update(budget(),dict(kind='test'))
    compare_exact(before,trainer._update_boundary());assert trainer._checkpoint_safe and not trainer.fp32_events


def test_decoder_precision_policy_and_inactive_state_remain_unchanged(staged):
    joint=StagedTrainer(staged,'plan.json','joint',local_check=True);joint.update();joint.save(staged/'joint.pt')
    trainer=ExceptionTrainer(staged,'plan.json','decoder_only',local_check=True,handoff=staged/'joint.pt',context=context(staged))
    assert trainer.scaler_initial_scale==1024 and trainer.scaler_max_scale==1024
    trainer.update(budget());trainer.verify_inactive();assert not trainer.fp32_events
    with pytest.raises(ValueError,match='Only joint'):trainer.fp32_update(budget(),{})


def test_nonfinite_adam_is_rejected():
    with pytest.raises(ValueError,match='Nonfinite accepted'):finite_tree({'adam':[torch.tensor(float('inf'))]})


def test_import_inactive_anchor_uses_immutable_checkpoint_not_fresh_initialization(staged):
    trainer=ExceptionTrainer(staged,'plan.json','joint',local_check=True,context=context(staged))
    state=dict(model=copy.deepcopy(trainer.model.state_dict()))
    key=next(iter(trainer.inactive));state['model'][key]=state['model'][key]+.001
    trainer.model.load_state_dict(state['model'])
    with pytest.raises(AssertionError):trainer.verify_inactive()
    anchor_inactive(trainer,state);trainer.verify_inactive()
    with torch.no_grad():trainer.model.state_dict()[key].add_(.001)
    with pytest.raises(AssertionError):trainer.verify_inactive()


def test_budget_rollback_clears_pending_accepted_inference(staged,monkeypatch):
    import json
    trainer=ExceptionTrainer(staged,'plan.json','joint',local_check=True,context=context(staged))
    before=trainer._update_boundary()
    monkeypatch.setattr(StagedTrainer,'update',lambda self,allocation:None)
    assert trainer.fp32_update(budget(),dict(kind='test')) is None
    compare_exact(before,trainer._update_boundary());assert not trainer.fp32_events
    log=json.loads((staged/'events/fp32-exception-events.json').read_text())
    assert log['events']==[] and log['pending']['status']=='budget_rolled_back'


def test_cpu_boundary_import_does_not_alias_immutable_adam(staged):
    trainer=ExceptionTrainer(staged,'plan.json','joint',local_check=True,context=context(staged))
    trainer.update(budget());trainer.save(staged/'boundary.pt')
    state=torch.load(staged/'boundary.pt',weights_only=True);before=copy.deepcopy(state)
    load_boundary(trainer,state);trainer.fp32_update(budget(),dict(kind='test'))
    compare_exact(state,before)


def test_resume_preflight_rejects_exhausted_unfinished_joint(tmp_path):
    import json
    from scripts.recover_staged_tokenizer_fp32 import validate_resume_report
    from scripts.run_tokenizer_comparison import source_hash
    from genie_diep.data import file_hash
    manifest_path=tmp_path/'manifest.json';manifest_path.write_text('{}');sha=file_hash(manifest_path)
    manifest=dict(original_plan_sha256='plan',original_budget_sha256='budget',parent_checkpoint_sha256='parent',failed_proof_sha256='failure')
    metadata=dict(manifest_sha256=sha,source_sha256=source_hash('scripts/recover_staged_tokenizer_fp32.py'),policy=POLICY)
    immutable=tmp_path/'verified.pt';immutable.write_bytes(b'verified')
    checkpoint=tmp_path/'joint.pt';torch.save(dict(step=2000,contract=dict(precision_exception=metadata),fp32_exception_events=[{}]*4),checkpoint)
    report=dict(identity=dict(mode='full',local_check=False,plan_sha256='plan',budget_sha256='budget'),
        fp32_recovery=dict(manifest_sha256=sha,parent_checkpoint_sha256='parent',failed_proof_sha256='failure'),
        phases=dict(joint=dict(latest_step=2000,checkpoint=dict(path=checkpoint.name,sha256=file_hash(checkpoint)))))
    (tmp_path/'experiment-report.json').write_text(json.dumps(report))
    proof=dict(status='passed',manifest_sha256=sha,source_sha256=metadata['source_sha256'],
        verified_checkpoint_path=immutable.name,checkpoint_sha256=file_hash(immutable))
    with pytest.raises(ValueError,match='review before allocation'):validate_resume_report(tmp_path,manifest_path,manifest,proof)
    state=torch.load(checkpoint,weights_only=True);state['fp32_exception_events'].pop();torch.save(state,checkpoint)
    report['phases']['joint']['checkpoint']['sha256']=file_hash(checkpoint)
    (tmp_path/'experiment-report.json').write_text(json.dumps(report))
    assert validate_resume_report(tmp_path,manifest_path,manifest,proof)==dict(joint=2000)
