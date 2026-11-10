"""Bounded baseline stages with checked data, exact CPU resume and gated promotion.

Local preflight updates verify execution only. They are never a trained model result.
"""
import argparse
import copy
import json
import hashlib
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .cohort import Cohort, MixedSampler, read
from .data import file_hash
from .model import Config, WorldModel, image_loss


def stage_config(spec, stage):return next(s for s in spec['stages'] if s['name']==stage)


def implementation_hash():
    digest=hashlib.sha256()
    for name in ['baseline.py','cohort.py','model.py','baseline_evaluation.py']:
        digest.update(name.encode());digest.update(Path(__file__).with_name(name).read_bytes().replace(b'\r\n',b'\n'))
    return digest.hexdigest()


def validate_readiness(root, specification):
    root=Path(root);spec=read(root/specification)
    readiness=read(root/spec['readiness'])
    if not readiness.get('ready_for_gpu_pilot') or readiness['implementation_sha256']!=implementation_hash():raise ValueError('Readiness evidence missing or stale')
    if readiness['spec_sha256']!=file_hash(root/specification) or readiness['binding_sha256']!=file_hash(root/spec['data_binding']):raise ValueError('Readiness configuration changed')
    return readiness


def dynamics_mask(ids, final_fraction=.5):
    if torch.rand(()).item()<final_fraction:
        mask=torch.zeros_like(ids,dtype=torch.bool);mask[:,-1]=True
    else:
        mask=torch.rand(ids.shape,device=ids.device)<float(torch.empty((),device=ids.device).uniform_(.5,1))
        mask[:,0]=False;mask[:,-1,0,0]=True
    return mask


def rng_state():
    n=np.random.get_state()
    return dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu());random.setstate(state['python'])
    n=state['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    if state['cuda']:torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda']])


class Budget:
    """Conservative elapsed-allocation accounting; export reserve is never training time."""
    def __init__(self, stage, prior_seconds=0, prior_units=0, rate=0, balance=None, setup_seconds=0, reserve_seconds=120):
        if rate<0 or setup_seconds<0:raise ValueError('Invalid allocation accounting')
        self.stage=stage;self.prior_seconds=prior_seconds;self.prior_units=prior_units
        self.rate=rate;self.balance=balance;self.setup=setup_seconds;self.reserve=reserve_seconds;self.started=time.monotonic()
        if balance is not None and balance<=50:raise ValueError('Monthly reserve would be crossed')
    def spent(self):
        elapsed=self.setup+time.monotonic()-self.started
        return self.prior_seconds+elapsed,self.prior_units+elapsed*self.rate/3600
    def remaining(self):
        seconds,units=self.spent();limits=[self.stage['max_allocated_hours']*3600-seconds]
        if self.rate:
            limits.append((self.stage['max_compute_units']-units)*3600/self.rate)
            if self.balance is not None:
                current_elapsed=self.setup+time.monotonic()-self.started
                limits.append((self.balance-50)*3600/self.rate-current_elapsed)
        return min(limits)-self.reserve
    def permits(self, expected_seconds=0):return self.remaining()>max(0,expected_seconds)


class Trainer:
    def __init__(self, root, specification, stage, device='cpu', precision='fp32', microbatch=None, accumulation=None,
                 preflight=False, resume=None, initial=None, promotion=None):
        self.root=Path(root).resolve();self.spec_path=self.root/specification;self.spec=read(self.spec_path)
        self.stage=stage;self.stage_spec=stage_config(self.spec,stage);self.device=torch.device(device)
        self.preflight=preflight;self.precision=precision
        if precision not in {'fp32','fp16'} or (precision=='fp16' and self.device.type!='cuda'):raise ValueError('Unsupported precision')
        if not preflight and not self.spec.get('local_execution_ready'):raise ValueError('Local readiness gate not passed')
        if not preflight:validate_readiness(self.root,specification)
        self.data=Cohort(self.root,self.spec['data_binding'],'train',2 if stage=='actions' else self.spec['model']['context'])
        self.validation=Cohort(self.root,self.spec['data_binding'],'val',self.spec['model']['context'])
        self.evaluation=read(self.root/self.spec['evaluation_selection'])
        if self.evaluation['binding_sha256']!=self.data.binding_sha256:raise ValueError('Evaluation cohort changed')
        self.sampler=MixedSampler(self.data,self.spec['optimization']['seed'],self.spec['data']['source_balanced_sampling_fraction'])
        nominal=self.stage_spec.get('microbatch_sequences',self.stage_spec.get('microbatch_pairs'))
        self.microbatch=microbatch or nominal;self.accumulation=accumulation or self.stage_spec['gradient_accumulation']
        if min(self.microbatch,self.accumulation)<1:raise ValueError('Invalid batch settings')
        if not preflight and self.microbatch*self.accumulation!=nominal*self.stage_spec['gradient_accumulation']:
            raise ValueError('Fallback must preserve effective batch size')
        seed=self.spec['optimization']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        self.model=WorldModel(Config(**self.spec['model'])).to(self.device)
        self.contract=dict(spec_sha256=file_hash(self.spec_path),binding_sha256=self.data.binding_sha256,
            evaluation_sha256=file_hash(self.root/self.spec['evaluation_selection']),stage=stage,precision=precision,
            microbatch=self.microbatch,accumulation=self.accumulation,preflight=preflight,torch_version=str(torch.__version__),
            implementation_sha256=implementation_hash(),device_type=self.device.type)
        self.step=0;self.best=float('inf');self.spent_seconds=0;self.spent_units=0
        state=None
        if resume:
            state=torch.load(resume,map_location='cpu',weights_only=True)
            if state['contract']!=self.contract:raise ValueError('Exact resume contract changed; create an explicit new experiment')
            self.model.load_state_dict(state['model'])
        elif initial:
            state_initial=torch.load(initial,map_location='cpu',weights_only=True)
            if not preflight:
                gate=read(promotion) if promotion else {}
                expected='G1' if stage=='actions' else 'G2'
                if gate.get('gate')!=expected or gate.get('status')!='pass' or gate.get('checkpoint_sha256')!=file_hash(initial):
                    raise ValueError('Required prior-stage promotion gate not passed')
                if state_initial['contract']['preflight']:raise ValueError('Cannot promote a preflight checkpoint')
                if state_initial['contract']['binding_sha256']!=self.data.binding_sha256:raise ValueError('Prior-stage cohort differs')
                if state_initial['config']!=self.model.config_dict():raise ValueError('Architecture changed')
            self.model.load_state_dict(state_initial['model'])
        elif stage!='tokenizer' and not preflight:raise ValueError('Prior-stage checkpoint and gate are required')
        if not resume:
            with torch.no_grad():
                x=self.sampler.batch(1).to(self.device)
                if stage=='tokenizer':self.model.tokenizer.vq.initialize(self.model.tokenizer.features(x))
                if stage=='actions':
                    f=self.model.lam.features(x)
                    for i,book in enumerate(self.model.lam.books):book.initialize(f[:,:,i])
        self.model.requires_grad_(False)
        self.module={'tokenizer':self.model.tokenizer,'actions':self.model.lam,'dynamics':self.model.dynamics}[stage]
        self.module.requires_grad_(True);self.model.eval();self.module.train()
        self.parameters=list(self.module.parameters())
        self.optimizer=torch.optim.AdamW(self.parameters,lr=self.stage_spec['learning_rate'],weight_decay=self.spec['optimization']['weight_decay'])
        maximum=self.stage_spec['max_updates'];warmup=500
        def rate(step):
            if step<warmup:return (step+1)/warmup
            progress=min(1,(step-warmup)/max(1,maximum-warmup));return .1+.9*(1+math.cos(math.pi*progress))/2
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,rate)
        self.scaler_initial_scale=float(self.spec['optimization'].get('fp16_initial_scale',1024))
        if not math.isfinite(self.scaler_initial_scale) or self.scaler_initial_scale<=0:
            raise ValueError('FP16 initial scale must be finite and positive')
        self.scaler_max_scale=float(self.spec['optimization'].get('fp16_max_scale',1024))
        if not math.isfinite(self.scaler_max_scale) or self.scaler_max_scale<self.scaler_initial_scale:
            raise ValueError('FP16 maximum scale must be finite and at least the initial scale')
        self.scaler=self._new_scaler()
        self.last_failure=None;self._checkpoint_safe=True
        if state:
            self.optimizer.load_state_dict(state['optimizer']);self.scheduler.load_state_dict(state['scheduler'])
            self.scaler.load_state_dict(state['scaler']);self.sampler.load_state_dict(state['sampler']);self.step=state['step']
            self.best=state['best'];self.spent_seconds=state['spent_seconds'];self.spent_units=state['spent_units'];restore_rng(state['rng'])
            self.last_failure=state.get('last_failure')

    def _new_scaler(self):
        return torch.amp.GradScaler('cuda',enabled=self.precision=='fp16',init_scale=self.scaler_initial_scale)

    def _advance_scaler(self):
        self.scaler.update()
        # Strict failure handling deliberately does not skip or retry a batch.
        # Bound automatic growth so a conservative start cannot quietly return
        # to the original 65,536 scale over a long run.
        if self.scaler.is_enabled() and self.scaler.get_scale()>self.scaler_max_scale:
            self.scaler.update(new_scale=self.scaler_max_scale)

    def _update_boundary(self):
        # Copies stay on their original device. This costs model + Adam-state memory,
        # but also makes exceptions during a partially completed optimizer step safe.
        return dict(model=copy.deepcopy(self.model.state_dict()),optimizer=copy.deepcopy(self.optimizer.state_dict()),
            scheduler=copy.deepcopy(self.scheduler.state_dict()),scaler=copy.deepcopy(self.scaler.state_dict()),
            rng=rng_state(),sampler=self.sampler.state_dict(),step=self.step)

    def _restore_update_boundary(self,boundary):
        self.model.load_state_dict(boundary['model']);self.optimizer.load_state_dict(boundary['optimizer'])
        self.scheduler.load_state_dict(boundary['scheduler'])
        # load_state_dict alone does not reset GradScaler's private per-optimizer
        # READY/UNSCALED state. A fresh scaler makes in-process recovery safe too.
        self.scaler=self._new_scaler();self.scaler.load_state_dict(boundary['scaler'])
        self.sampler.load_state_dict(boundary['sampler']);restore_rng(boundary['rng']);self.step=boundary['step']
        self.optimizer.zero_grad(set_to_none=True);self._checkpoint_safe=True

    def _failure_diagnostics(self,boundary,phase,microbatches,error):
        affected=[]
        for name,p in self.model.named_parameters():
            if p.grad is None:continue
            grad=p.grad.detach();finite=torch.isfinite(grad)
            if bool(finite.all()):continue
            values=grad[finite]
            affected.append(dict(name=name,elements=grad.numel(),nonfinite=int((~finite).sum()),
                nan=int(torch.isnan(grad).sum()),positive_inf=int(torch.isposinf(grad).sum()),
                negative_inf=int(torch.isneginf(grad).sum()),max_finite_abs=float(values.abs().max()) if values.numel() else None))
        replay=MixedSampler(self.data,source_fraction=self.sampler.fraction);replay.load_state_dict(boundary['sampler'])
        windows=[self.data.descriptor(*replay.draw()) for _ in range(self.sampler.draws-boundary['sampler']['draws'])]
        return dict(attempted_step=boundary['step']+1,phase=phase,completed_microbatches=microbatches,
            exception=type(error).__name__,message=str(error),precision=self.precision,scale=self.scaler.get_scale(),
            sampler_draws_before=boundary['sampler']['draws'],sampler_draws_at_failure=self.sampler.draws,
            sampled_windows=windows,affected_gradients=affected,restored_pre_update_boundary=False)

    def loss(self,x):
        if self.stage=='tokenizer':
            rec,_,vq=self.model.tokenizer(x);return 10*image_loss(rec,x,color_balance=self.stage_spec['color_balance'])+vq
        if self.stage=='actions':
            rec,_,_,vq=self.model.lam(x);return 10*image_loss(rec,x[:,1:],x[:,:-1])+vq
        with torch.no_grad():
            _,ids,_=self.model.tokenizer.encode(x);actions,_,_=self.model.lam.infer(x)
        mask=dynamics_mask(ids,self.stage_spec['fully_mask_final_frame_fraction'])
        logits=self.model.dynamics(ids.masked_fill(mask,self.model.config.vocab),actions.detach())
        return F.cross_entropy(logits[mask],ids[mask])

    def update(self,budget=None):
        self.last_failure=None
        if self.step>=self.stage_spec['max_updates']:raise ValueError('Update cap reached')
        self.optimizer.zero_grad(set_to_none=True);losses=[]
        boundary=self._update_boundary();self._checkpoint_safe=False;phase='sampling'
        try:
            for _ in range(self.accumulation):
                if budget is not None and not budget.permits():
                    self._restore_update_boundary(boundary);return None
                phase='sampling';x=self.sampler.batch(self.microbatch).to(self.device)
                phase='forward'
                with torch.autocast(device_type=self.device.type,dtype=torch.float16,enabled=self.precision=='fp16'):
                    loss=self.loss(x)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; no optimizer update')
                phase='backward';self.scaler.scale(loss/self.accumulation).backward();losses.append(float(loss.detach()))
            phase='unscale';self.scaler.unscale_(self.optimizer)
            phase='gradient_check'
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in self.parameters):
                raise FloatingPointError('Nonfinite gradient; no optimizer update')
            phase='gradient_clip'
            torch.nn.utils.clip_grad_norm_(self.parameters,self.spec['optimization']['gradient_clip_norm'],error_if_nonfinite=True)
            phase='optimizer_step';self.scaler.step(self.optimizer)
            phase='scaler_update';self._advance_scaler()
            phase='scheduler_step';self.scheduler.step();self.step+=1;self._checkpoint_safe=True
        except BaseException as error:
            try:self.last_failure=self._failure_diagnostics(boundary,phase,len(losses),error)
            except Exception as diagnostic_error:
                self.last_failure=dict(attempted_step=boundary['step']+1,phase=phase,exception=type(error).__name__,
                    message=str(error),diagnostic_error=str(diagnostic_error),restored_pre_update_boundary=False)
            self._restore_update_boundary(boundary)
            self.last_failure['restored_pre_update_boundary']=True
            raise
        return sum(losses)/len(losses)

    @torch.no_grad()
    def evaluate(self,limit=None,budget=None):
        self.model.eval();records=[];estimate=0
        for item in self.evaluation['validation'][:limit]:
            if budget is not None and not budget.permits(estimate):break
            started=time.monotonic()
            x=self.validation.window(*self.validation.locate(item))[None].to(self.device)
            if self.stage=='tokenizer':
                prediction,_,_=self.model.tokenizer(x);target=x
                error=(prediction-target).square();metric=float(error.mean())
            elif self.stage=='actions':
                x=x[:,-2:];prediction,_,_,_=self.model.lam(x);target=x[:,1:];metric=float((prediction-target).square().mean())
            else:
                _,ids,_=self.model.tokenizer.encode(x);actions,_,_=self.model.lam.infer(x);masked=ids.clone();masked[:,-1]=self.model.config.vocab
                logits=self.model.dynamics(masked,actions)[:,-1];metric=float(F.cross_entropy(logits.reshape(-1,self.model.config.vocab),ids[:,-1].flatten()))
            records.append(dict(source_id=item['source_id'],value=metric))
            estimate=max(estimate,(time.monotonic()-started)*1.25)
        self.module.train()
        by_source={s:[r['value'] for r in records if r['source_id']==s] for s in sorted({r['source_id'] for r in records})}
        return dict(stage=self.stage,step=self.step,samples=len(records),sources=len(by_source),
            pooled=float(np.mean([r['value'] for r in records])) if records else None,
            equal_source=float(np.mean([np.mean(v) for v in by_source.values()])) if by_source else None,
            by_source={s:float(np.mean(v)) for s,v in by_source.items()},
            interpretation='Reconstruction diagnostic' if self.stage=='tokenizer' else 'Future-inferred oracle diagnostic; not commanded control',
            quality_gate='unmeasured',preflight=self.preflight,complete=len(records)==len(self.evaluation['validation'][:limit]))

    def save(self,path,budget=None):
        if not self._checkpoint_safe:raise RuntimeError('Cannot save an incomplete update: rollback did not finish')
        if budget:self.spent_seconds,self.spent_units=budget.spent()
        state=dict(schema_version=1,config=self.model.config_dict(),contract=self.contract,model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict(),scaler=self.scaler.state_dict(),
            sampler=self.sampler.state_dict(),rng=rng_state(),step=self.step,best=self.best,
            spent_seconds=self.spent_seconds,spent_units=self.spent_units,last_failure=self.last_failure)
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
        torch.save(state,temporary);temporary.replace(path)


def run(args):
    torch.set_num_threads(args.threads)
    out=Path(args.out)
    if out.exists() and not args.resume:raise FileExistsError('Use a new experiment directory')
    if args.device.startswith('cuda') and (args.rate<=0 or args.balance is None or args.allocation_start is None):
        raise ValueError('GPU execution requires a live rate, balance and allocation start timestamp')
    if not args.preflight and not args.authorized:raise ValueError('Explicit training authorization required')
    if not args.preflight and args.device.startswith('cuda'):
        pilot=read(args.pilot_report) if args.pilot_report else {}
        if pilot.get('status')!='passed' or pilot.get('implementation_sha256')!=implementation_hash():raise ValueError('Passing GPU pilot required')
        spec=read(Path(args.root)/args.spec)
        if pilot.get('spec_sha256')!=file_hash(Path(args.root)/args.spec) or pilot.get('binding_sha256')!=file_hash(Path(args.root)/spec['data_binding']):raise ValueError('GPU pilot inputs changed')
    trainer=Trainer(args.root,args.spec,args.stage,args.device,args.precision,args.microbatch,args.accumulation,
                    args.preflight,args.resume,args.initial,args.promotion)
    setup=max(0,time.time()-args.allocation_start) if args.allocation_start else 0
    stage=dict(trainer.stage_spec)
    if args.preflight:stage.update(max_allocated_hours=min(stage['max_allocated_hours'],.25),max_compute_units=min(stage['max_compute_units'],2))
    budget=Budget(stage,trainer.spent_seconds,trainer.spent_units,args.rate,args.balance,setup,120 if args.device.startswith('cuda') else 0)
    out.mkdir(parents=True,exist_ok=True)
    (out/'run.json').write_text(json.dumps(dict(contract=trainer.contract,arguments=vars(args),purpose='execution_check_only' if args.preflight else 'bounded_experiment'),indent=2)+'\n',encoding='utf-8')
    end=min(args.updates or stage['max_updates'],stage['max_updates']);last_save=last_eval=time.monotonic();estimate=0
    if args.preflight:end=min(end,trainer.step+3)
    try:
        while trainer.step<end and budget.permits(estimate):
            started=time.monotonic();loss=trainer.update(budget)
            if loss is None:break
            estimate=max(estimate,(time.monotonic()-started)*1.25)
            print(json.dumps(dict(step=trainer.step,loss=loss,remaining_seconds=budget.remaining())),flush=True)
            now=time.monotonic()
            if trainer.step%500==0 or now-last_eval>=900:
                if not budget.permits(estimate*4):break
                report=trainer.evaluate(budget=budget);last_eval=time.monotonic()
                with (out/'evaluation.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(report)+'\n')
                if report['complete'] and report['equal_source']<trainer.best:trainer.best=report['equal_source'];trainer.save(out/'best.pt',budget)
            if now-last_save>=300:trainer.save(out/'last.pt',budget);last_save=time.monotonic()
    except BaseException as error:
        failure=trainer.last_failure or dict(exception=type(error).__name__,message=str(error),phase='outside_update')
        (out/'failure.json').write_text(json.dumps(failure,indent=2)+'\n',encoding='utf-8')
        raise
    finally:trainer.save(out/'last.pt',budget)
    print(json.dumps(dict(checkpoint=str(out/'last.pt'),step=trainer.step,preflight=args.preflight,promotion='not_approved')),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.');p.add_argument('--spec',default='configs/training-baseline-v1.json')
    p.add_argument('--stage',choices=['tokenizer','actions','dynamics'],required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--precision',choices=['fp32','fp16'],default='fp32');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--microbatch',type=int);p.add_argument('--accumulation',type=int);p.add_argument('--updates',type=int)
    p.add_argument('--preflight',action='store_true');p.add_argument('--authorized',action='store_true');p.add_argument('--resume');p.add_argument('--initial');p.add_argument('--promotion')
    p.add_argument('--pilot-report')
    p.add_argument('--rate',type=float,default=0);p.add_argument('--balance',type=float);p.add_argument('--allocation-start',type=float)
    run(p.parse_args())


if __name__=='__main__':main()
