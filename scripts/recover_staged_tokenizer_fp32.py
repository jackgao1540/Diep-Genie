"""Versioned exceptional joint FP32 updates; sealed AMP gates remain unchanged.

Verify starts at original1803, compares an independent full FP32 AdamW reference,
replays the exceptional update, retains six unique updates, and returns. Resume
requires that proof. Four FP32 events per lineage force a safe review pause.
"""
import argparse
import copy
import gc
import math
from pathlib import Path
import shutil
import time

import torch
from genie_diep.baseline import rng_state, restore_rng
from genie_diep.cohort import read
from genie_diep.data import file_hash
from scripts import run_staged_tokenizer as original
from scripts import recover_staged_tokenizer as prior
from scripts.run_tokenizer_comparison import compare_exact, source_hash

POLICY = dict(version='staged16-fp32-exception-v1', maximum_joint_events=4,
    initial_exception_update=1804, original_amp_scale=1024., preserve_amp_growth_tracker=True,
    recurrent_eligibility='finite-forward-AMP-gradient_check-overflow-after-exact-rollback',
    stageB_precision_policy='unchanged_original', verification_maximum_seconds=900,
    verification_maximum_units=.3, minimum_operation_seconds=60,
    old_fp16_gradient_norm_relative_error_limit=.05)
KEYS = ('model','optimizer','scheduler','scaler','sampler','rng','step')
require, write_json = prior.require, prior.write_json


def cpu_copy(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_copy(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def finite_tree(value):
    if isinstance(value,torch.Tensor):
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()),'Nonfinite accepted state')
    elif isinstance(value,dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value,(list,tuple)):
        for v in value: finite_tree(v)


def convert_state(state,contract):
    result=copy.deepcopy(state);result['contract']=copy.deepcopy(contract)
    result['fp32_exception_events']=[]
    for key in state:
        if key!='contract':compare_exact(result[key],state[key])
    return result


def load_boundary(trainer,state):
    if state['sampler'].get('schema')=='train-frame-coverage-v1':trainer._attach_coverage(reconstruct=True)
    trainer.model.load_state_dict(state['model'])
    # CPU load_state_dict may alias input Adam tensors; the immutable source
    # must also survive a CPU reference or sequential replay unchanged.
    trainer.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    trainer.scheduler.load_state_dict(state['scheduler'])
    trainer.scaler=trainer._new_scaler();trainer.scaler.load_state_dict(state['scaler'])
    trainer.sampler.load_state_dict(state['sampler']);trainer.step=state['step']
    trainer.best=state.get('best',float('inf'));trainer.spent_seconds=state.get('spent_seconds',0)
    trainer.spent_units=state.get('spent_units',0);trainer.last_failure=state.get('last_failure')
    trainer._checkpoint_safe=True;restore_rng(state['rng'])


def anchor_inactive(trainer,state):
    """Use immutable imported tensors, never a fresh device initialization."""
    trainer.inactive={key:state['model'][key].to(trainer.device).clone() for key in trainer.inactive}


class ExceptionTrainer(original.StagedTrainer):
    def __init__(self,root,plan_path,phase,device='cpu',microbatch=4,handoff=None,resume=None,
                 local_check=False,mode='pilot',*,context):
        super().__init__(root,plan_path,phase,device,microbatch,handoff,None,local_check,mode)
        self.context=context;self.fp32_events=[]
        self.contract['precision_exception']=dict(manifest_sha256=context['manifest_sha256'],
            source_sha256=source_hash(__file__),policy=POLICY)
        if resume:
            state=torch.load(resume,map_location='cpu',weights_only=True)
            require(state['contract']==self.contract,'Exact FP32-exception contract changed')
            load_boundary(self,state);self.fp32_events=state.get('fp32_exception_events',[])
        self.inactive={k:v.detach().clone() for k,v in self.model.state_dict().items()
            if not k.startswith('tokenizer.') or (phase=='decoder_only' and k.startswith(('tokenizer.encoder.','tokenizer.vq.')))}

    _failure_diagnostics=prior.RecoveryTrainer._failure_diagnostics

    def _event_log(self,event):
        write_json(Path(self.context['out'])/'fp32-exception-events.json',dict(
            events=self.fp32_events,pending=event,manifest_sha256=self.context['manifest_sha256']))

    def fp32_update(self,budget,reason):
        require(self.phase=='joint','Only joint FP32 exceptions are authorized')
        require(len(self.fp32_events)<POLICY['maximum_joint_events'],'FP32 event cap reached; safe review required')
        require(budget is not None and budget.permits(POLICY['minimum_operation_seconds']),
                'Insufficient budget for an exceptional FP32 update')
        before=self._update_boundary();amp=copy.deepcopy(before['scaler']);precision=self.precision
        sampler=copy.deepcopy(before['sampler']);random=before['rng']
        try:
            descriptors=[]
            for _ in range(self.microbatch*self.accumulation):
                descriptors.append(self.data.descriptor(*self.sampler.draw()))
        finally:self.sampler.load_state_dict(sampler);restore_rng(random)
        event=dict(accepted_step_before=self.step,attempted_update=self.step+1,precision='fp32',
            reason=reason,sampled_windows=descriptors,amp_state_preserved=True,dropped_samples=0,status='pending')
        self._event_log(event);started=time.monotonic()
        try:
            self.precision='fp32';self.scaler=torch.amp.GradScaler('cuda',enabled=False)
            loss=super().update(budget)
            if loss is None:
                self._event_log(dict(event,status='budget_rolled_back'))
                return None
            finite_tree(self.model.state_dict());finite_tree(self.optimizer.state_dict());self.verify_inactive()
            require(self.step==before['step']+1,'FP32 exception changed update count')
        except BaseException:
            self.precision=precision;self._restore_update_boundary(before)
            self._event_log(dict(event,status='failed_rolled_back'));raise
        finally:
            self.precision=precision;self.scaler=self._new_scaler();self.scaler.load_state_dict(amp)
        event.update(status='accepted',accepted_step_after=self.step,loss=loss,seconds=time.monotonic()-started)
        self.fp32_events.append(event);self._event_log(None)
        return loss

    def update(self,budget=None):
        if self.phase!='joint':return super().update(budget)
        require(len(self.fp32_events)<POLICY['maximum_joint_events'],'FP32 event cap reached; safe review required')
        if self.step==self.plan['coverage_after_update']:self._attach_coverage()
        if self.step==POLICY['initial_exception_update']-1:
            return self.fp32_update(budget,dict(kind='bound_failed_T4_FP16_comparison',
                evidence_sha256=self.context['failed_proof_sha256']))
        before=self._update_boundary()
        try:return super().update(budget)
        except FloatingPointError:
            failure=copy.deepcopy(self.last_failure)
            require(self.precision=='fp16' and failure and failure.get('phase')=='gradient_check' and
                failure.get('message')=='Nonfinite gradient; no optimizer update' and
                failure.get('restored_pre_update_boundary') is True and self._checkpoint_safe,
                'Not an eligible AMP gradient overflow; stop safely')
            actual=self._update_boundary()
            for key in KEYS:compare_exact(actual[key],before[key])
            return self.fp32_update(budget,dict(kind='recurrent_AMP_gradient_overflow',failure=failure,
                exact_rollback=True))

    def save(self,path,budget=None):
        require(self._checkpoint_safe,'Cannot save unsafe FP32-exception boundary')
        if budget:self.spent_seconds,self.spent_units=budget.spent()
        state=dict(schema_version=1,config=self.model.config_dict(),contract=self.contract,
            model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict(),
            scaler=self.scaler.state_dict(),sampler=self.sampler.state_dict(),rng=rng_state(),step=self.step,best=self.best,
            spent_seconds=self.spent_seconds,spent_units=self.spent_units,last_failure=self.last_failure,
            fp32_exception_events=self.fp32_events)
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
        torch.save(state,temporary);temporary.replace(path)


def reference_update(trainer,budget):
    """Independent explicit FP32 reference: original loss, clip, Adam and schedule."""
    require(budget.permits(60),'Insufficient reference budget')
    trainer.optimizer.zero_grad(set_to_none=True);losses=[]
    for _ in range(trainer.accumulation):
        require(budget.permits(),'Reference budget exhausted')
        x=trainer.sampler.batch(trainer.microbatch).to(trainer.device)
        with torch.autocast(trainer.device.type,enabled=False):loss=trainer.loss(x.float())
        require(bool(torch.isfinite(loss)),'Nonfinite FP32 reference loss')
        (loss/trainer.accumulation).backward();losses.append(float(loss.detach()))
    require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in trainer.parameters),'Nonfinite FP32 reference gradient')
    norm=float(torch.nn.utils.clip_grad_norm_(trainer.parameters,trainer.spec['optimization']['gradient_clip_norm'],error_if_nonfinite=True))
    trainer.optimizer.step();trainer.scheduler.step();trainer.step+=1
    finite_tree(trainer.model.state_dict());finite_tree(trainer.optimizer.state_dict());trainer.verify_inactive()
    return dict(loss=sum(losses)/len(losses),preclip_gradient_norm=norm,
        boundary=cpu_copy(trainer._update_boundary()))


def exact_replay(trainer,folder,budget):
    """Restore events as well as model/Adam/RNG; old generic replay omits events."""
    before=folder/'replay-before.pt';trainer.save(before)
    started=time.monotonic();loss=trainer.update(budget);require(loss is not None,'Replay budget exhausted')
    if trainer.device.type=='cuda':torch.cuda.synchronize()
    seconds=time.monotonic()-started;after=folder/'replay-after.pt';trainer.save(after)
    expected=torch.load(after,map_location='cpu',weights_only=True)
    state=torch.load(before,map_location='cpu',weights_only=True)
    load_boundary(trainer,state);trainer.fp32_events=copy.deepcopy(state['fp32_exception_events'])
    replay_loss=trainer.update(budget);require(replay_loss==loss,'FP32-exception replay loss changed')
    actual=cpu_copy(trainer._update_boundary())
    for key in KEYS:compare_exact(actual[key],expected[key])
    # Wall durations are measurements, not scientific event identity.
    def events(value):return [{k:v for k,v in e.items() if k!='seconds'} for e in value]
    compare_exact(events(trainer.fp32_events),events(expected['fp32_exception_events']))
    trainer.save(folder/'replay-actual.pt')
    return dict(passed=True,seconds=seconds,loss=loss,maximum_state_difference=0,unique_updates=1,extra_optimizer_calls=1)


def validate_manifest(root,path):
    manifest=read(path)
    require(manifest['schema_version']==1 and manifest['policy']==POLICY,'FP32-exception policy changed')
    require(manifest['source_sha256']==source_hash(__file__),'FP32-exception source changed')
    require(manifest['prior_wrapper_sha256']==source_hash(prior.__file__),'Supporting wrapper changed')
    require(manifest['original_plan_sha256']==file_hash(Path(root)/manifest['plan']),'Original plan changed')
    original.validate_plan(root,manifest['plan'])
    require(manifest['accepted_step']==1803 and manifest['failed_update']==1804,'FP32 boundary changed')
    return manifest


def context_for(path,manifest,out):
    return dict(manifest_sha256=file_hash(path),failed_proof_sha256=manifest['failed_proof_sha256'],out=str(out))


def verify(args,manifest,budget):
    root=Path(args.root).resolve();source=Path(args.source_run).resolve()
    out=original.validate_output(root,args.out,args.device,args.private_output_root)
    require(not out.exists(),'Fresh FP32 verification branch required')
    paths=[(source/'experiment-report.json','parent_report_sha256'),(source/'joint/step-001803.pt','parent_checkpoint_sha256'),
        (source/'joint/initial.pt','parent_initial_sha256'),(Path(args.failed_proof),'failed_proof_sha256')]
    for path,key in paths:require(file_hash(path)==manifest[key],'Immutable parent/evidence changed')
    failed=read(args.failed_proof);cases=failed['cases'];report=read(paths[0][0])
    require(failed['status']=='failed' and failed['error']['message']=='Lower-scale FP16 gradient norm differs beyond original bound',
            'Wrong rejected AMP proof')
    require(cases['fp32']['status']=='finite' and cases['fp16_1024']['status']=='nonfinite' and
        cases['fp16_256']['status']=='finite' and
        abs(cases['fp16_256']['unscaled_gradient_norm']/cases['fp32']['unscaled_gradient_norm']-1)>.05,
        'Evidence does not establish rejected AMP comparison')
    require(report['status']=='stopped' and report['identity']['mode']=='full' and report['phases']['joint']['latest_step']==1803,
        'Not original stopped full1803')
    state=torch.load(paths[1][0],map_location='cpu',weights_only=True)
    require(state['step']==1803 and state['scaler']['scale']==1024 and state['scaler']['_growth_tracker']==1803,'Original AMP identity changed')
    finite_tree(state['model']);finite_tree(state['optimizer'])
    out.mkdir(parents=True);folder=out/'joint';folder.mkdir()
    proof=dict(status='running',manifest_sha256=file_hash(args.recovery_manifest),source_sha256=source_hash(__file__),
        parent_checkpoint_sha256=manifest['parent_checkpoint_sha256'],failed_proof_sha256=manifest['failed_proof_sha256'],
        policy=POLICY,accepted_step_before=1803,stageB_unchanged=True,
        gpu=torch.cuda.get_device_name(),torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
        backend_policy=original.BACKEND_POLICY,original_amp_state=state['scaler'])
    proof_path=out/'fp32-verification.json';write_json(proof_path,proof)
    # Sequential models: retain reference on CPU and release CUDA before candidate.
    reference=original.StagedTrainer(root,args.plan,'joint',args.device,4,resume=paths[1][0],mode='full')
    expected=reference_update(reference,budget)
    del reference;gc.collect();torch.cuda.empty_cache()
    context=context_for(args.recovery_manifest,manifest,out)
    trainer=ExceptionTrainer(root,args.plan,'joint',args.device,4,mode='full',context=context)
    migrated=convert_state(state,trainer.contract);torch.save(migrated,out/'migrated-boundary.pt')
    initial=torch.load(paths[2][0],map_location='cpu',weights_only=True)
    torch.save(convert_state(initial,trainer.contract),folder/'initial.pt')
    load_boundary(trainer,migrated)
    anchor_inactive(trainer,state)
    loss=trainer.update(budget);require(loss==expected['loss'],'FP32 reference loss differs')
    actual=cpu_copy(trainer._update_boundary())
    for key in KEYS:compare_exact(actual[key],expected['boundary'][key])
    proof['reference']=dict(exact=True,loss=loss,preclip_gradient_norm=expected['preclip_gradient_norm'],
        compared=list(KEYS),amp_state_unchanged=True)
    write_json(proof_path,proof)
    del expected,actual;gc.collect()
    # Replay the exceptional update itself from1803, retaining only replay1804.
    load_boundary(trainer,migrated);trainer.fp32_events=[]
    proof['replay']=exact_replay(trainer,folder,budget)
    write_json(proof_path,proof)
    proof['timing']=original.measure_training_updates(trainer,budget,5)
    trainer.verify_inactive();final=folder/f'step-{trainer.step:06d}.pt';trainer.save(final,budget)
    require(trainer.step==1809,'Expected six retained unique updates')
    immutable=out/'verified-step-001809.pt';shutil.copyfile(final,immutable)
    entry=report['phases']['joint'];entry['prior_saved_checkpoints']=entry.pop('saved_checkpoints',{})
    entry.update(initial_checkpoint_sha256=file_hash(folder/'initial.pt'),latest_step=1809,
        checkpoint=dict(path=final.relative_to(out).as_posix(),sha256=file_hash(final)),
        saved_checkpoints={'1809':dict(path=final.relative_to(out).as_posix(),sha256=file_hash(final))},
        recovery_timing=proof['timing'])
    report['status']='fp32_recovery_verified';report.pop('error',None)
    report['fp32_recovery']=dict(manifest_sha256=file_hash(args.recovery_manifest),
        parent_checkpoint_sha256=manifest['parent_checkpoint_sha256'],failed_proof_sha256=manifest['failed_proof_sha256'])
    report['spent_seconds'],report['spent_units']=budget.spent();write_json(out/'experiment-report.json',report)
    for path,key in paths:require(file_hash(path)==manifest[key],'Immutable evidence changed during verification')
    proof.update(status='passed',accepted_step_after=1809,verified_checkpoint_path=immutable.name,
        checkpoint_sha256=file_hash(final),report_sha256=file_hash(out/'experiment-report.json'),
        initial_adapter_sha256=file_hash(folder/'initial.pt'),migrated_boundary_sha256=file_hash(out/'migrated-boundary.pt'),
        fp32_exception_events=trainer.fp32_events,final_amp_state=trainer.scaler.state_dict(),
        spent_seconds=report['spent_seconds'],spent_units=report['spent_units'])
    write_json(proof_path,proof);return proof


def validate_resume_report(out,manifest_path,manifest,proof):
    out=Path(out).resolve();sha=file_hash(manifest_path)
    require(proof['status']=='passed' and proof['manifest_sha256']==sha and proof['source_sha256']==source_hash(__file__),
        'Exact completed FP32 CUDA proof required')
    immutable=(out/proof['verified_checkpoint_path']).resolve()
    require(immutable.is_relative_to(out) and file_hash(immutable)==proof['checkpoint_sha256'],'Immutable FP32 checkpoint changed')
    report=read(out/'experiment-report.json');identity=report['identity'];lineage=report['fp32_recovery']
    require(identity['mode']=='full' and identity['local_check'] is False and identity['plan_sha256']==manifest['original_plan_sha256'] and
        identity['budget_sha256']==manifest['original_budget_sha256'],'Progressive full identity changed')
    require(lineage['manifest_sha256']==sha and lineage['parent_checkpoint_sha256']==manifest['parent_checkpoint_sha256'] and
        lineage['failed_proof_sha256']==manifest['failed_proof_sha256'],'Progressive FP32 lineage changed')
    steps={}
    for phase,entry in report['phases'].items():
        if 'checkpoint' not in entry:continue
        path=(out/entry['checkpoint']['path']).resolve()
        require(path.is_relative_to(out) and file_hash(path)==entry['checkpoint']['sha256'],'Progressive checkpoint changed')
        state=torch.load(path,map_location='cpu',weights_only=True);metadata=state['contract']['precision_exception']
        require(state['step']==entry['latest_step'] and metadata==dict(manifest_sha256=sha,source_sha256=source_hash(__file__),policy=POLICY),
            'Progressive state/FP32 contract changed')
        require(len(state['fp32_exception_events'])<=POLICY['maximum_joint_events'],'Event cap changed')
        require(phase!='joint' or state['step']>=5000 or len(state['fp32_exception_events'])<POLICY['maximum_joint_events'],
            'FP32 event cap exhausted for unfinished joint phase; review before allocation')
        steps[phase]=state['step']
    return steps


def run(args):
    original.configure_backend();torch.set_num_threads(args.threads)
    require(args.device=='cuda' and args.authorized and torch.cuda.is_available(),'Authorized CUDA only')
    manifest=validate_manifest(args.root,args.recovery_manifest)
    require(args.plan==manifest['plan'] and file_hash(args.full_budget)==manifest['original_budget_sha256'],'Plan/budget changed')
    mode=args.mode;args.mode='full';args.local_check=False
    budget,_=original.allocation_budget(args);plan=original.validate_plan(args.root,args.plan)
    if mode=='verify':
        original.full_pilot_gate(args.root,args,plan,budget,{'joint':1803})
        budget.stage=dict(max_allocated_hours=min(26,(args.prior_seconds+900)/3600),max_compute_units=min(30,args.prior_units+.3))
        try:return verify(args,manifest,budget)
        except BaseException as error:
            path=Path(args.out)/'fp32-verification.json'
            if path.exists():
                proof=read(path);proof.update(status='failed',error=dict(type=type(error).__name__,message=str(error)));write_json(path,proof)
            raise
        finally:args.mode=mode
    proof=read(Path(args.out)/'fp32-verification.json');validate_resume_report(args.out,args.recovery_manifest,manifest,proof)
    context=context_for(args.recovery_manifest,manifest,args.out);original_class=original.StagedTrainer
    original.StagedTrainer=lambda *a,**kw:ExceptionTrainer(*a,**kw,context=context);args.resume=True
    try:return original.run(args)
    finally:original.StagedTrainer=original_class;args.mode=mode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('verify','resume'),required=True);p.add_argument('--root',required=True)
    p.add_argument('--plan',required=True);p.add_argument('--recovery-manifest',required=True)
    p.add_argument('--source-run',required=True);p.add_argument('--failed-proof',required=True);p.add_argument('--out',required=True)
    p.add_argument('--private-output-root');p.add_argument('--full-budget',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--authorized',action='store_true')
    p.add_argument('--rate',type=float,default=0);p.add_argument('--balance',type=float)
    p.add_argument('--allocation-start',type=float);p.add_argument('--prior-seconds',type=float,default=0)
    p.add_argument('--prior-units',type=float,default=0);p.add_argument('--threads',type=int,default=2)
    run(p.parse_args())

if __name__=='__main__':main()
