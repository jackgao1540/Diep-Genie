"""Versioned precision-only staged continuation; original artifacts stay immutable.

Verify mode must reproduce the exact old-scale gradient failure, pass same-batch
FP32/lower-FP16 comparisons and exact saved-update replay, then return. Resume
mode requires that proof. Joint-only retries never skip samples or change Adam,
LR, architecture, objective, batch grouping, coverage or global budget limits.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import time

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch

from genie_diep.baseline import Budget, restore_rng, rng_state
from genie_diep.cohort import MixedSampler, read
from genie_diep.coverage_sampler import CoverageSampler, sampler_state_sha256
from genie_diep.data import file_hash
from genie_diep.gpu_precision import _backward_case
from scripts.diagnose_tokenizer_precision import gradient_report
from scripts.run_tokenizer_comparison import compare_exact, source_hash
from scripts import run_staged_tokenizer as original


POLICY = dict(version='staged16-transactional-scale-backoff-v1', trial_scales=[1024,512,256,128],
    minimum_scale=128, maximum_backoffs_per_update=2, factor=.5,
    reset_growth_tracker_on_backoff=True, joint_maximum_scale='largest_verified_lower_scale',
    stageB_precision_policy='unchanged_original', maximum_gradient_norm_relative_error=.05,
    verification_maximum_seconds=900, verification_maximum_units=.3)


def require(condition, message):
    if not condition: raise ValueError(message)


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp'); temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8'); temporary.replace(path)


def validate_manifest(root, path):
    manifest=read(path)
    require(manifest['schema_version']==1 and manifest['policy']==POLICY,'Recovery policy changed')
    require(manifest['source_sha256']==source_hash(__file__),'Recovery source changed')
    require(manifest['original_plan_sha256']==file_hash(Path(root)/manifest['plan']),'Original plan changed')
    original.validate_plan(root,manifest['plan'])
    require(manifest['accepted_step']==1803 and manifest['failed_update']==1804,'Recovery boundary changed')
    return manifest


def recover_sampler(data, state, plan, spec):
    """Restore the saved safe boundary, not the legacy consumed-batch rewind."""
    mixed=MixedSampler(data,spec['optimization']['seed'],spec['data']['source_balanced_sampling_fraction'])
    if state['sampler'].get('schema')=='train-frame-coverage-v1':
        for _ in range(1+plan['coverage_after_update']*16): mixed.draw()
        parent=mixed.state_dict()
        sampler=CoverageSampler(data,parent,seed=plan['coverage_seed'],expected_parent_sampler_sha256=sampler_state_sha256(parent))
    else: sampler=mixed
    sampler.load_state_dict(state['sampler']); compare_exact(sampler.state_dict(),state['sampler'])
    require(sampler.draws==1+state['step']*16,'Saved sampler is not the accepted-update boundary')
    return sampler


def next_batches(trainer):
    state=copy.deepcopy(trainer.sampler.state_dict()); random=rng_state()
    try: return [trainer.sampler.batch(trainer.microbatch) for _ in range(trainer.accumulation)]
    finally: trainer.sampler.load_state_dict(state); restore_rng(random)


def backward(trainer, batches, precision, scale, budget):
    sampler=copy.deepcopy(trainer.sampler.state_dict()); random=rng_state()
    try:
        return _backward_case(trainer.model,batches,precision,scale,lambda model,x:trainer.loss(x),budget)
    finally:
        trainer.optimizer.zero_grad(set_to_none=True);trainer.sampler.load_state_dict(sampler);restore_rng(random)


def scaler_backoff_state(saved, selected):
    result=copy.deepcopy(saved)
    require(saved and math.isfinite(float(saved['scale'])) and 0<selected<saved['scale'],'Invalid loss-scale backoff')
    require(saved['backoff_factor']==.5 and selected==saved['scale']*.5,'Backoff must be a standard factor of two')
    result['scale']=float(selected);result['_growth_tracker']=0
    return result


def converted_state(state, contract, selected_scale=None):
    """Lossless explicit provenance adapter; only scaler precision metadata varies."""
    result=copy.deepcopy(state); result['contract']=copy.deepcopy(contract)
    if selected_scale is not None:
        old=float(state['scaler']['scale']);require(selected_scale in (512,256,128) and selected_scale<old,'Unverified recovery scale')
        while old>selected_scale:
            result['scaler']=scaler_backoff_state(result['scaler'],old*.5);old*=.5
    for key in ('model','optimizer','scheduler','sampler','rng','step','best','spent_seconds','spent_units','last_failure','config'):
        compare_exact(result[key],state[key])
    return result


class RecoveryTrainer(original.StagedTrainer):
    def __init__(self,root,plan_path,phase,device='cpu',microbatch=4,handoff=None,resume=None,
                 local_check=False,mode='pilot',*,context):
        super().__init__(root,plan_path,phase,device,microbatch,handoff,None,local_check,mode)
        self.context=context; self.backoff_events=[]
        self.contract['precision_recovery']=dict(manifest_sha256=context['manifest_sha256'],
            source_sha256=source_hash(__file__),policy=POLICY,selected_joint_maximum_scale=context['selected_scale'])
        if phase=='joint':
            self.scaler_initial_scale=float(context['selected_scale']);self.scaler_max_scale=float(context['selected_scale'])
            self.scaler=self._new_scaler()
        if resume:
            state=torch.load(resume,map_location='cpu',weights_only=True)
            require(state['contract']==self.contract,'Exact recovery continuation contract changed')
            if state['sampler'].get('schema')=='train-frame-coverage-v1':self._attach_coverage(reconstruct=True)
            self.model.load_state_dict(state['model']);self.optimizer.load_state_dict(state['optimizer']);self.scheduler.load_state_dict(state['scheduler'])
            self.scaler.load_state_dict(state['scaler']);self.sampler.load_state_dict(state['sampler']);self.step=state['step'];self.best=state['best']
            self.spent_seconds=state['spent_seconds'];self.spent_units=state['spent_units'];self.last_failure=state.get('last_failure');restore_rng(state['rng'])
            self.backoff_events=state.get('precision_backoff_events',[])
        self.inactive={k:v.detach().clone() for k,v in self.model.state_dict().items() if not k.startswith('tokenizer.') or
            (phase=='decoder_only' and k.startswith(('tokenizer.encoder.','tokenizer.vq.')))}

    def _failure_diagnostics(self,boundary,phase,microbatches,error):
        # The original MixedSampler-only reporter loses affected tensors on a
        # coverage failure. This reporter does not change the transaction itself.
        if isinstance(self.sampler,CoverageSampler):
            parent=self.sampler._parent_state
            replay=CoverageSampler(self.data,parent,seed=self.plan['coverage_seed'],expected_parent_sampler_sha256=sampler_state_sha256(parent))
        else: replay=MixedSampler(self.data,source_fraction=self.sampler.fraction)
        replay.load_state_dict(boundary['sampler'])
        samples=[self.data.descriptor(*replay.draw()) for _ in range(self.sampler.draws-boundary['sampler']['draws'])]
        gradients=gradient_report(self.model,1.)
        return dict(attempted_step=boundary['step']+1,phase=phase,completed_microbatches=microbatches,
            exception=type(error).__name__,message=str(error),scale=self.scaler.get_scale(),sampled_windows=samples,
            affected_gradients=[g for g in gradients if g['nonfinite']],gradient_values='after baseline unscale at gradient_check',
            restored_pre_update_boundary=False)

    def update(self,budget=None):
        if self.phase!='joint':return super().update(budget)
        if self.step==self.plan['coverage_after_update']:self._attach_coverage()
        before=self._update_boundary()
        for attempt in range(POLICY['maximum_backoffs_per_update']+1):
            try:return super().update(budget)
            except FloatingPointError:
                failure=copy.deepcopy(self.last_failure)
                eligible=(self.precision=='fp16' and failure and failure.get('phase')=='gradient_check' and
                          failure.get('restored_pre_update_boundary') is True and self._checkpoint_safe)
                if not eligible:raise
                # Scaler changes are the only permitted differences from the
                # first failed boundary between retries of this same update.
                actual=self._update_boundary()
                for key in ('model','optimizer','scheduler','sampler','rng','step'):compare_exact(actual[key],before[key])
                current=float(self.scaler.get_scale()); lower=current*.5
                if attempt>=POLICY['maximum_backoffs_per_update'] or lower<POLICY['minimum_scale']:raise
                require(budget is not None and budget.permits(60),'Insufficient budget for same-batch FP32 overflow diagnosis')
                batches=next_batches(self);case=backward(self,batches,'fp32',1.,budget)
                require(case['status']=='finite' and case['model_state_unchanged'],'FP32 failed: do not classify this as scale-only overflow')
                lower_case=backward(self,batches,'fp16',lower,budget)
                require(lower_case['status']=='finite' and lower_case['model_state_unchanged'],
                        'Proposed recurrent backoff is not finite; stop without accepting or dropping a batch')
                relative=abs(lower_case['unscaled_gradient_norm']/case['unscaled_gradient_norm']-1)
                require(relative<=POLICY['maximum_gradient_norm_relative_error'],'Recurrent lower-scale gradient differs beyond bound')
                saved=self.scaler.state_dict();self.scaler=self._new_scaler();self.scaler.load_state_dict(scaler_backoff_state(saved,lower))
                self.backoff_events.append(dict(accepted_step=self.step,attempted_update=self.step+1,from_scale=current,to_scale=lower,
                    growth_tracker_reset=0,exact_state_rollback=True,dropped_samples=0,fp32_case=case,
                    lower_fp16_case=lower_case,gradient_norm_relative_error=relative,original_failure=failure))
                write_json(Path(self.context['out'])/'precision-backoff-events.json',self.backoff_events)

    def save(self,path,budget=None):
        require(self._checkpoint_safe,'Cannot save an unsafe recovery boundary')
        if budget:self.spent_seconds,self.spent_units=budget.spent()
        state=dict(schema_version=1,config=self.model.config_dict(),contract=self.contract,model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict(),scaler=self.scaler.state_dict(),
            sampler=self.sampler.state_dict(),rng=rng_state(),step=self.step,best=self.best,spent_seconds=self.spent_seconds,
            spent_units=self.spent_units,last_failure=self.last_failure,precision_backoff_events=self.backoff_events)
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(path)


def recovery_context(manifest_path,selected,out):
    return dict(manifest_sha256=file_hash(manifest_path),selected_scale=float(selected),out=str(out))


def verify(args,manifest,budget):
    root=Path(args.root).resolve();out=original.validate_output(root,args.out,args.device,args.private_output_root)
    source=Path(args.source_run).resolve();require(not out.exists(),'Recovery verification needs a fresh output branch')
    report_path=source/'experiment-report.json';checkpoint=source/'joint/step-001803.pt';initial_path=source/'joint/initial.pt'
    for path,key in ((report_path,'parent_report_sha256'),(checkpoint,'parent_checkpoint_sha256'),(initial_path,'parent_initial_sha256')):
        require(file_hash(path)==manifest[key],'Original recovery artifact changed: '+path.name)
    source_report=read(report_path)
    require(source_report['status']=='stopped' and source_report['identity']['mode']=='full' and
            source_report['identity']['local_check'] is False and source_report['phases']['joint']['latest_step']==1803 and
            source_report['identity']['budget_sha256']==manifest['original_budget_sha256'], 'Not the bound stopped full experiment')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    require(state['step']==1803 and state['last_failure']['phase']=='gradient_check' and state['last_failure']['restored_pre_update_boundary'],'Not the verified safe failed boundary')
    require(state['scaler']['scale']==1024 and state['scaler']['_growth_tracker']==1803,'Original scaler identity changed')
    # Load exactly under the original contract before any new provenance/scale.
    trainer=original.StagedTrainer(root,args.plan,'joint',args.device,4,resume=checkpoint,mode='full')
    before=trainer._update_boundary();batches=next_batches(trainer);cases={}
    out.mkdir(parents=True)
    diagnostic=dict(status='diagnostic_running',cases=cases,manifest_sha256=file_hash(args.recovery_manifest),
        parent_checkpoint_sha256=manifest['parent_checkpoint_sha256'],source_sha256=source_hash(__file__))
    write_json(out/'recovery-verification.json',diagnostic)
    cases['fp32']=backward(trainer,batches,'fp32',1.,budget)
    write_json(out/'recovery-verification.json',diagnostic)
    require(cases['fp32']['status']=='finite','Same-batch CUDA FP32 is not finite')
    cases['fp16_1024']=backward(trainer,batches,'fp16',1024.,budget)
    write_json(out/'recovery-verification.json',diagnostic)
    old=cases['fp16_1024'];require(old['status']=='nonfinite' and all(m['finite_loss'] for m in old['microbatches']),
        'Original finite-forward gradient overflow did not reproduce; investigate exact state instead')
    selected=None;relative=None
    for scale in POLICY['trial_scales'][1:]:
        case=cases['fp16_'+str(scale)]=backward(trainer,batches,'fp16',float(scale),budget)
        write_json(out/'recovery-verification.json',diagnostic)
        if case['status']=='finite':
            relative=abs(case['unscaled_gradient_norm']/cases['fp32']['unscaled_gradient_norm']-1)
            require(relative<=POLICY['maximum_gradient_norm_relative_error'],'Lower-scale FP16 gradient norm differs beyond original bound')
            selected=scale;break
        require(case['status']=='nonfinite' and all(m['finite_loss'] for m in case['microbatches']),
                'Lower precision case failed outside finite-forward gradient overflow')
    require(selected is not None,'No bound lower scale is finite; do not resume')
    compare_exact(trainer._update_boundary(),before)
    folder=out/'joint';folder.mkdir()
    context=recovery_context(args.recovery_manifest,selected,out)
    recovered=RecoveryTrainer(root,args.plan,'joint',args.device,4,mode='full',context=context)
    boundary=converted_state(state,recovered.contract,selected)
    migrated=out/'migrated-boundary.pt';torch.save(boundary,migrated)
    initial_adapter=converted_state(initial,recovered.contract,None);torch.save(initial_adapter,folder/'initial.pt')
    recovered=RecoveryTrainer(root,args.plan,'joint',args.device,4,resume=migrated,mode='full',context=context)
    proof=dict(status='running',manifest_sha256=file_hash(args.recovery_manifest),parent_checkpoint_sha256=manifest['parent_checkpoint_sha256'],
        source_sha256=source_hash(__file__),selected_scale=selected,maximum_joint_scale=selected,growth_tracker_after_conversion=0,
        stageB_unchanged=True,policy=POLICY,cases=cases,gradient_norm_relative_error=relative,
        exact_state_conversion_except_contract_scaler=True,accepted_step_before=1803,precision_diagnostic_optimizer_updates=0)
    write_json(out/'recovery-verification.json',proof)
    proof['replay']=original.replay_check(recovered,folder,budget)
    proof['timing']=original.measure_training_updates(recovered,budget,5)
    recovered.verify_inactive();final=folder/f'step-{recovered.step:06d}.pt';recovered.save(final,budget)
    require(recovered.step==1809,'Unexpected recovery accepted-update count')
    immutable=out/'verified-step-001809.pt';shutil.copyfile(final,immutable)
    report=read(report_path);report['status']='recovery_verified';report.pop('error',None)
    entry=report['phases']['joint'];entry['prior_saved_checkpoints']=entry.pop('saved_checkpoints',{})
    entry.update(initial_checkpoint_sha256=file_hash(folder/'initial.pt'),latest_step=recovered.step,
        checkpoint=dict(path=final.relative_to(out).as_posix(),sha256=file_hash(final)),
        saved_checkpoints={str(recovered.step):dict(path=final.relative_to(out).as_posix(),sha256=file_hash(final))},
        recovery_timing=proof['timing'])
    report['recovery']=dict(original_run=str(source),manifest_sha256=file_hash(args.recovery_manifest),selected_scale=selected,
        parent_checkpoint_sha256=manifest['parent_checkpoint_sha256'],initial_model_optimizer_unchanged=True)
    report['spent_seconds'],report['spent_units']=budget.spent();write_json(out/'experiment-report.json',report)
    proof.update(status='passed',accepted_step_after=recovered.step,checkpoint_sha256=file_hash(final),
        verified_checkpoint_path=immutable.name,
        migrated_boundary_sha256=file_hash(migrated),initial_adapter_sha256=file_hash(folder/'initial.pt'),
        report_sha256=file_hash(out/'experiment-report.json'),spent_seconds=report['spent_seconds'],spent_units=report['spent_units'])
    write_json(out/'recovery-verification.json',proof)
    for path,key in ((report_path,'parent_report_sha256'),(checkpoint,'parent_checkpoint_sha256'),(initial_path,'parent_initial_sha256')):
        require(file_hash(path)==manifest[key],'Original artifact changed during verification')
    return proof


def validate_resume_report(out,manifest_path,manifest,proof):
    """Validate immutable proof and current progressive endpoints independently.

    The working step1809 file can legitimately be rewritten by a later safe
    stop with newer spending metadata. The separate verification copy cannot.
    """
    out=Path(out).resolve();manifest_sha=file_hash(manifest_path)
    require(proof['status']=='passed' and proof['manifest_sha256']==manifest_sha and
            proof['source_sha256']==source_hash(__file__),'Exact completed CUDA recovery verification required')
    immutable=(out/proof['verified_checkpoint_path']).resolve()
    require(immutable.is_relative_to(out) and file_hash(immutable)==proof['checkpoint_sha256'],
            'Immutable verified recovery checkpoint changed')
    report=read(out/'experiment-report.json');identity=report['identity'];lineage=report['recovery']
    require(identity['mode']=='full' and identity['local_check'] is False and
            identity['plan_sha256']==manifest['original_plan_sha256'] and
            identity['budget_sha256']==manifest['original_budget_sha256'],'Progressive full-run identity changed')
    require(lineage['manifest_sha256']==manifest_sha and lineage['selected_scale']==proof['selected_scale'] and
            lineage['parent_checkpoint_sha256']==manifest['parent_checkpoint_sha256'],'Progressive recovery lineage changed')
    steps={}
    for phase,entry in report['phases'].items():
        if 'checkpoint' not in entry:continue
        path=(out/entry['checkpoint']['path']).resolve()
        require(path.is_relative_to(out) and file_hash(path)==entry['checkpoint']['sha256'],'Progressive checkpoint changed')
        state=torch.load(path,map_location='cpu',weights_only=True)
        require(state['step']==entry['latest_step'],'Progressive report step differs from actual checkpoint')
        recovery=state['contract']['precision_recovery']
        require(recovery['manifest_sha256']==manifest_sha and recovery['source_sha256']==source_hash(__file__) and
                recovery['selected_joint_maximum_scale']==proof['selected_scale'],'Progressive checkpoint recovery contract changed')
        steps[phase]=state['step']
    return steps


def run(args):
    original.configure_backend();torch.set_num_threads(args.threads)
    require(args.device=='cuda' and args.authorized and torch.cuda.is_available(),'Authorized CUDA only; use local tests/CPU diagnosis separately')
    manifest=validate_manifest(args.root,args.recovery_manifest)
    require(args.plan==manifest['plan'],'Plan path changed')
    require(file_hash(args.full_budget)==manifest['original_budget_sha256'],'Original budget receipt changed')
    # The existing allocation and pilot conditions remain required even for the
    # separately bounded verification allocation.
    args.mode_saved=args.mode;args.mode='full';args.local_check=False
    budget,limits=original.allocation_budget(args)
    plan=original.validate_plan(args.root,args.plan)
    if args.mode_saved=='verify':
        original.full_pilot_gate(args.root,args,plan,budget,{'joint':1803})
        budget.stage=dict(max_allocated_hours=min(26,(args.prior_seconds+900)/3600),max_compute_units=min(30,args.prior_units+.3))
        try:return verify(args,manifest,budget)
        except BaseException as error:
            path=Path(args.out)/'recovery-verification.json'
            if path.exists():
                proof=read(path);proof.update(status='failed',error=dict(type=type(error).__name__,message=str(error)))
                write_json(path,proof)
            raise
        finally:args.mode=args.mode_saved
    proof=read(Path(args.out)/'recovery-verification.json')
    validate_resume_report(args.out,args.recovery_manifest,manifest,proof)
    context=recovery_context(args.recovery_manifest,proof['selected_scale'],args.out)
    original_class=original.StagedTrainer
    original.StagedTrainer=lambda *a,**kw:RecoveryTrainer(*a,**kw,context=context)
    args.resume=True
    try:return original.run(args)
    finally:original.StagedTrainer=original_class;args.mode=args.mode_saved


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('verify','resume'),required=True);p.add_argument('--root',required=True)
    p.add_argument('--plan',default='configs/staged-tokenizer-run-v1.json');p.add_argument('--recovery-manifest',required=True)
    p.add_argument('--source-run',required=True);p.add_argument('--out',required=True);p.add_argument('--private-output-root')
    p.add_argument('--full-budget',required=True);p.add_argument('--device',default='cuda');p.add_argument('--authorized',action='store_true')
    p.add_argument('--rate',type=float,required=True);p.add_argument('--balance',type=float,required=True);p.add_argument('--allocation-start',type=float,required=True)
    p.add_argument('--prior-seconds',type=float,required=True);p.add_argument('--prior-units',type=float,required=True);p.add_argument('--threads',type=int,default=2)
    result=run(p.parse_args());print(json.dumps(dict(status=result.get('status')),indent=2))


if __name__=='__main__':main()
