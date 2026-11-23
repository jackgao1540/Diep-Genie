import copy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.recover_staged_tokenizer import (
    RecoveryTrainer, converted_state, scaler_backoff_state, recover_sampler, POLICY,
    validate_resume_report, run,
)
from scripts.run_staged_tokenizer import StagedTrainer
from scripts.run_tokenizer_comparison import compare_exact
from test_staged_tokenizer_runner import staged
from test_baseline_pipeline import project
from genie_diep.data import file_hash
from scripts.run_tokenizer_comparison import source_hash


def test_standard_backoff_resets_growth_only_and_rejects_nonstandard_jump():
    before=dict(scale=1024.,growth_factor=2.,backoff_factor=.5,growth_interval=2000,_growth_tracker=1803)
    after=scaler_backoff_state(before,512)
    assert after==dict(before,scale=512.,_growth_tracker=0) and before['_growth_tracker']==1803
    with pytest.raises(ValueError,match='factor of two'):scaler_backoff_state(before,256)


def test_conversion_preserves_complete_accepted_state(staged):
    trainer=StagedTrainer(staged,'plan.json','joint',local_check=True,mode='full');trainer.update();trainer.save(staged/'old.pt')
    state=torch.load(staged/'old.pt',weights_only=True)
    state['scaler']=dict(scale=1024.,growth_factor=2.,backoff_factor=.5,growth_interval=2000,_growth_tracker=1803)
    original=copy.deepcopy(state);contract=dict(state['contract'],precision_recovery='test')
    result=converted_state(state,contract,256)
    assert result['scaler']['scale']==256 and result['scaler']['_growth_tracker']==0
    for key in state:
        if key not in ('contract','scaler'):compare_exact(result[key],state[key])
    compare_exact(state,original)


def test_recovery_loader_exact_replay_and_changed_contract_rejected(staged):
    context=dict(manifest_sha256='test-manifest',selected_scale=256,out=str(staged/'events'))
    trainer=RecoveryTrainer(staged,'plan.json','joint',local_check=True,mode='full',context=context)
    trainer.update();trainer.save(staged/'boundary.pt');loss=trainer.update()
    resumed=RecoveryTrainer(staged,'plan.json','joint',local_check=True,mode='full',resume=staged/'boundary.pt',context=context)
    assert resumed.update()==loss
    compare_exact(resumed._update_boundary(),trainer._update_boundary())
    with pytest.raises(ValueError,match='contract changed'):
        RecoveryTrainer(staged,'plan.json','joint',local_check=True,mode='full',resume=staged/'boundary.pt',
                        context=dict(context,selected_scale=128))


def test_coverage_failure_diagnostics_keep_actual_gradient_name_and_windows(staged):
    context=dict(manifest_sha256='test-manifest',selected_scale=256,out=str(staged/'events'))
    trainer=RecoveryTrainer(staged,'plan.json','joint',local_check=True,mode='full',context=context)
    trainer._attach_coverage();before=trainer._update_boundary()
    trainer.sampler.batch(4)
    parameter=next(trainer.module.encoder.parameters());parameter.grad=torch.full_like(parameter,float('inf'))
    report=trainer._failure_diagnostics(before,'gradient_check',1,FloatingPointError('injected'))
    assert len(report['sampled_windows'])==4
    assert report['affected_gradients'] and report['affected_gradients'][0]['name'].startswith('tokenizer.encoder.')
    assert report['affected_gradients'][0]['nonfinite']==parameter.numel()


def test_stageB_scaler_policy_unchanged(staged):
    joint=StagedTrainer(staged,'plan.json','joint',local_check=True);joint.update();joint.save(staged/'joint.pt')
    context=dict(manifest_sha256='test-manifest',selected_scale=256,out=str(staged/'events'))
    decoder=RecoveryTrainer(staged,'plan.json','decoder_only',handoff=staged/'joint.pt',local_check=True,context=context)
    assert decoder.scaler_initial_scale==1024 and decoder.scaler_max_scale==1024
    assert {id(p) for p in decoder.parameters}=={id(p) for p in decoder.module.decoder.parameters()}
    decoder.update();decoder.verify_inactive()


def late_resume(tmp_path):
    manifest_path=tmp_path/'manifest.json';manifest_path.write_text('{}')
    manifest=dict(original_plan_sha256='plan',original_budget_sha256='budget',parent_checkpoint_sha256='parent')
    metadata=dict(manifest_sha256=file_hash(manifest_path),source_sha256=source_hash('scripts/recover_staged_tokenizer.py'),selected_joint_maximum_scale=256.)
    immutable=tmp_path/'verified-step-001809.pt';immutable.write_bytes(b'original verified boundary')
    phases={}
    for phase,step in [('joint',5000),('decoder_only',3000)]:
        path=tmp_path/(phase+'.pt');torch.save(dict(step=step,contract=dict(precision_recovery=metadata)),path)
        phases[phase]=dict(latest_step=step,checkpoint=dict(path=path.name,sha256=file_hash(path)))
    report=dict(identity=dict(mode='full',local_check=False,plan_sha256='plan',budget_sha256='budget'),
        recovery=dict(manifest_sha256=file_hash(manifest_path),selected_scale=256.,parent_checkpoint_sha256='parent'),phases=phases)
    (tmp_path/'experiment-report.json').write_text(json.dumps(report))
    proof=dict(status='passed',manifest_sha256=file_hash(manifest_path),source_sha256=metadata['source_sha256'],
        selected_scale=256.,verified_checkpoint_path=immutable.name,checkpoint_sha256=file_hash(immutable))
    (tmp_path/'recovery-verification.json').write_text(json.dumps(proof))
    return manifest_path,manifest,proof


def test_late_progress_uses_actual_current_steps_and_immutable_replay_anchor(tmp_path):
    path,manifest,proof=late_resume(tmp_path)
    # A working filename may be rewritten; it must not invalidate the immutable
    # accepted verification copy or force a new forecast from step1803.
    (tmp_path/'joint').mkdir();(tmp_path/'joint/step-001809.pt').write_bytes(b'later safe-stop spending metadata')
    assert validate_resume_report(tmp_path,path,manifest,proof)==dict(joint=5000,decoder_only=3000)
    report=json.loads((tmp_path/'experiment-report.json').read_text());report['phases']['decoder_only']['latest_step']=4000
    (tmp_path/'experiment-report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='step differs'):validate_resume_report(tmp_path,path,manifest,proof)


def test_resume_delegates_progressive_forecast_instead_of_fixed1803(tmp_path,monkeypatch):
    from scripts import recover_staged_tokenizer as recovery
    path,manifest,proof=late_resume(tmp_path)
    budget_path=tmp_path/'budget.json';budget_path.write_text('{}');manifest['original_budget_sha256']=file_hash(budget_path)
    report=json.loads((tmp_path/'experiment-report.json').read_text());report['identity']['budget_sha256']=manifest['original_budget_sha256']
    (tmp_path/'experiment-report.json').write_text(json.dumps(report))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(recovery,'validate_manifest',lambda root,path:manifest)
    monkeypatch.setattr(recovery.original,'validate_plan',lambda *a:{})
    monkeypatch.setattr(recovery.original,'allocation_budget',lambda args:(None,None))
    def forbidden(*a,**kw):raise AssertionError('stale fixed1803 forecast must not run on resume')
    monkeypatch.setattr(recovery.original,'full_pilot_gate',forbidden)
    def progressive(args):
        assert args.mode=='full' and args.resume is True
        return dict(status='delegated-progressive-forecast')
    monkeypatch.setattr(recovery.original,'run',progressive)
    args=SimpleNamespace(device='cuda',authorized=True,threads=2,root=str(tmp_path),plan='plan',recovery_manifest=str(path),
        full_budget=str(budget_path),mode='resume',out=str(tmp_path))
    manifest['plan']='plan'
    assert run(args)['status']=='delegated-progressive-forecast'
