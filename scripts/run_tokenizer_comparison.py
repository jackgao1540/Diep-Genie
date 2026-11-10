"""Matched fresh tokenizer arms under one allocation and a strict paired boundary."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from genie_diep.baseline import Budget, Trainer, implementation_hash, restore_rng, rng_state, stage_config
from genie_diep.cohort import read
from genie_diep.data import file_hash
from genie_diep.model import image_loss


ARMS=('baseline','detail')


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def check(condition,message):
    if not condition:raise ValueError(message)


def compare_exact(actual,expected):
    if isinstance(expected,torch.Tensor):
        torch.testing.assert_close(actual.detach().cpu(),expected.detach().cpu(),rtol=0,atol=0)
    elif isinstance(expected,dict):
        check(actual.keys()==expected.keys(),'State keys changed')
        for key in expected:compare_exact(actual[key],expected[key])
    elif isinstance(expected,(tuple,list)):
        check(len(actual)==len(expected),'State length changed')
        for a,b in zip(actual,expected):compare_exact(a,b)
    else:check(actual==expected,'Scalar state changed')


def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def validate_experiment(root,experiment_path,baseline_spec,detail_spec):
    root=Path(root);path=root/experiment_path;experiment=read(path)
    sources=experiment['source_hashes']
    check(sources['implementation_sha256']==implementation_hash(),'Baseline implementation changed')
    for key,name in [('model_sha256','genie_diep/model.py'),('detail_loss_sha256','genie_diep/detail_loss.py'),
                     ('runner_sha256','scripts/run_tokenizer_comparison.py')]:
        check(sources[key]==source_hash(root/name),f'Comparison source changed: {name}')
    check(set(experiment['arms'])==set(ARMS),'Exactly baseline and detail arms are required')
    specs=[]
    for arm,filename in zip(ARMS,(baseline_spec,detail_spec)):
        entry=experiment['arms'][arm]
        check((root/entry['spec']).resolve()==(root/filename).resolve(),f'{arm} specification path differs')
        check(file_hash(root/filename)==entry['spec_sha256'],f'{arm} specification changed')
        spec=read(root/filename);specs.append(spec)
        if 'comparison_objective' in spec:
            check(spec['comparison_objective']==entry['objective'],f'{arm} objective metadata disagrees')
        check(stage_config(spec,'tokenizer')['max_updates']==experiment['schedule_max_updates'],
              'Comparison must preserve the baseline schedule horizon')
    # Readiness evidence and objective metadata may differ; all training/data
    # settings must otherwise match, including initialization and sampling.
    a,b=copy.deepcopy(specs)
    for spec in (a,b):
        for name in ('readiness','comparison_objective'):spec.pop(name,None)
    check(a==b,'Arm specifications differ outside objective/readiness')
    reference_path=root/experiment['reference_spec']
    check(file_hash(reference_path)==experiment['reference_spec_sha256'],'Reference baseline specification changed')
    reference=read(reference_path)
    for name in ('readiness','comparison_objective'):reference.pop(name,None)
    check(a==reference,'Baseline arm differs from the bound original specification')
    check(experiment['schedule_max_updates']==20000,'Preserve the existing 20,000-update learning-rate schedule')
    check(experiment['arms']['baseline']['objective']=={'name':'baseline'},'Baseline objective changed')
    objective=experiment['arms']['detail']['objective']
    check(objective['name']=='baseline_plus_detail','Unknown detail objective')
    check(math.isfinite(objective['weight']) and objective['weight']>0,'Detail coefficient must be positive and finite')
    check(set(objective)<= {'name','weight','loss_spec'},'Unknown detail objective setting')
    limits=experiment['limits']
    check(limits==dict(optimizer_calls_per_arm=1000,unique_updates_per_arm=999,maximum_compute_units=2,maximum_allocated_hours=1,minimum_remaining_units=50),
          'Comparison authorization limits changed')
    check(experiment['save_steps']==[250,500,750,999],'Common saved boundaries changed')
    return experiment


class ComparisonTrainer(Trainer):
    """An exact contract includes the objective code and bound comparison plan."""
    def __init__(self,root,specification,experiment_path,arm,device='cpu',preflight=False,resume=None):
        check(arm in ARMS,'Unknown comparison arm')
        root=Path(root);experiment=read(root/experiment_path)
        experiment=validate_experiment(root,experiment_path,experiment['arms']['baseline']['spec'],experiment['arms']['detail']['spec'])
        entry=experiment['arms'][arm]
        check((root/specification).resolve()==(root/entry['spec']).resolve(),'Arm specification mismatch')
        # Normal readiness checks still apply to every production-eligible arm.
        super().__init__(root,specification,'tokenizer',device,'fp16' if torch.device(device).type=='cuda' else 'fp32',
                         preflight=preflight)
        self.arm=arm;self.objective=copy.deepcopy(entry['objective'])
        self.contract['comparison']=dict(experiment_sha256=file_hash(root/experiment_path),arm=arm,
            objective=self.objective,source_hashes=experiment['source_hashes'])
        if resume:
            state=torch.load(resume,map_location='cpu',weights_only=True)
            check(state['contract']==self.contract,'Exact comparison resume contract changed')
            self.model.load_state_dict(state['model']);self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler']);self.scaler.load_state_dict(state['scaler'])
            self.sampler.load_state_dict(state['sampler']);self.step=state['step'];self.best=state['best']
            self.spent_seconds=state['spent_seconds'];self.spent_units=state['spent_units']
            self.last_failure=state.get('last_failure');restore_rng(state['rng'])
        self.arm_rng=rng_state()

    def loss(self,x):
        reconstruction,_,vq=self.model.tokenizer(x)
        value=10*image_loss(reconstruction,x,color_balance=self.stage_spec['color_balance'])+vq
        if self.arm=='detail':
            from genie_diep.detail_loss import detail_loss
            value=value+self.objective['weight']*detail_loss(reconstruction,x,self.objective.get('loss_spec'))
        return value

    def update(self,budget=None):
        restore_rng(self.arm_rng)
        try:return super().update(budget)
        finally:self.arm_rng=rng_state()

    def snapshot(self):
        restore_rng(self.arm_rng)
        return self._update_boundary()

    def restore_boundary(self,boundary):
        self._restore_update_boundary(boundary);self.arm_rng=rng_state()

    def save(self,path,budget=None):
        restore_rng(self.arm_rng)
        return super().save(path,budget)

    def evaluate(self,limit=None,budget=None):
        restore_rng(self.arm_rng)
        try:return super().evaluate(limit,budget)
        finally:self.arm_rng=rng_state()


def paired_update(trainers,budget,on_success=None,on_partial=None):
    """Commit both arms together; preserve an interrupted half-pair separately."""
    common=trainers['baseline'].step
    check(trainers['detail'].step==common,'Arms must begin at a common boundary')
    compare_exact(trainers['baseline'].sampler.state_dict(),trainers['detail'].sampler.state_dict())
    boundaries={arm:trainers[arm].snapshot() for arm in ARMS};results={}
    order=ARMS if common%2==0 else tuple(reversed(ARMS))
    try:
        for arm in order:
            started=time.monotonic();loss=trainers[arm].update(budget)
            if loss is None:raise TimeoutError('Allocation ended during a paired update')
            if trainers[arm].device.type=='cuda':torch.cuda.synchronize()
            results[arm]=dict(step=trainers[arm].step,loss=loss,seconds=time.monotonic()-started,
                              scale=trainers[arm].scaler.get_scale())
            if on_success:on_success(arm,results[arm])
        compare_exact(trainers['baseline'].sampler.state_dict(),trainers['detail'].sampler.state_dict())
        check(trainers['baseline'].step==trainers['detail'].step,'Completed pair has mismatched steps')
        return results
    except BaseException:
        try:
            for arm in ARMS:
                if trainers[arm].step>common and on_partial:on_partial(arm,trainers[arm])
        finally:
            for arm in ARMS:
                trainers[arm]._checkpoint_safe=False
                trainers[arm].restore_boundary(boundaries[arm])
        raise


def allocation_budget(args):
    if args.preflight:
        check(torch.device(args.device).type=='cpu','Local preflight must remain CPU-only')
        return Budget(dict(max_allocated_hours=1,max_compute_units=2),reserve_seconds=0)
    check(args.authorized,'Explicit paired-experiment authorization required')
    check(torch.device(args.device).type=='cuda','Production comparison requires authorized CUDA')
    now=time.time()
    values=(args.rate,args.balance,args.allocation_start)
    check(all(value is not None and math.isfinite(value) for value in values),'Live allocation accounting required')
    check(args.rate>0 and args.balance>50 and 0<args.allocation_start<=now,'Invalid live rate/balance/allocation start')
    return Budget(dict(max_allocated_hours=1,max_compute_units=2),rate=args.rate,balance=args.balance,
                  setup_seconds=now-args.allocation_start,reserve_seconds=180)


def run(args):
    root=Path(args.root).resolve();out=Path(args.out).resolve()
    check(not out.exists(),'Use a new comparison directory; existing experiments are immutable')
    budget=allocation_budget(args)
    experiment=validate_experiment(root,args.experiment,args.baseline_spec,args.detail_spec)
    maximum=2 if args.preflight else experiment['limits']['unique_updates_per_arm']
    target=args.updates if args.updates is not None else maximum
    check(2<=target<=maximum,'Requested update count exceeds the comparison/preflight limit')
    torch.set_num_threads(args.threads)
    if torch.device(args.device).type=='cuda':
        check(torch.cuda.is_available(),'CUDA unavailable');torch.cuda.reset_peak_memory_stats()
    out.mkdir(parents=True)
    report=dict(schema_version=1,status='running',preflight=args.preflight,promotion_approved=False,quality_gates='unmeasured',
        experiment_sha256=file_hash(root/args.experiment),source_hashes=experiment['source_hashes'],
        requested_unique_updates_per_arm=target,latest_common_step=0,optimizer_calls={arm:0 for arm in ARMS},
        replay_optimizer_calls={arm:0 for arm in ARMS},measurement={},saved_common_steps=[],
        allocation_accounting='One shared allocation; checkpoint usage fields are shared totals and must not be added across arms.',
        limits=experiment['limits'],device=args.device,torch_version=str(torch.__version__))
    trainers={};frozen=None;specifications=dict(zip(ARMS,(args.baseline_spec,args.detail_spec)))
    def write_report():
        report['spent_seconds'],report['estimated_units']=budget.spent()
        temp=out/'comparison-report.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        temp.replace(out/'comparison-report.json')
    def save_common(name):
        for arm in ARMS:trainers[arm].save(out/arm/name,budget)
    def on_success(arm,result):report['optimizer_calls'][arm]+=1
    def on_partial(arm,trainer):
        path=out/arm/f'partial-step-{trainer.step}.pt';trainer.save(path,budget)
        report.setdefault('partial_checkpoints',[]).append(dict(arm=arm,step=trainer.step,path=path.relative_to(out).as_posix(),
                                                              comparable=False))
    def evaluate_common(step):
        complete=True
        for arm in ARMS:
            diagnostic=trainers[arm].evaluate(budget=budget)
            with (out/arm/'evaluation.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(diagnostic)+'\n')
            report.setdefault('evaluations',[]).append(dict(arm=arm,common_step=step,report=diagnostic))
            complete=complete and diagnostic['complete'];write_report()
        if not complete:raise TimeoutError('Fixed validation interrupted by combined allocation budget')
    try:
        if not budget.permits(120):raise TimeoutError('Setup exhausted comparison budget')
        for arm in ARMS:
            trainers[arm]=ComparisonTrainer(root,specifications[arm],args.experiment,arm,args.device,args.preflight)
        compare_exact(trainers['baseline'].model.state_dict(),trainers['detail'].model.state_dict())
        compare_exact(trainers['baseline'].snapshot()['rng'],trainers['detail'].snapshot()['rng'])
        compare_exact(trainers['baseline'].sampler.state_dict(),trainers['detail'].sampler.state_dict())
        if not args.preflight:
            for trainer in trainers.values():check(len(trainer.evaluation['validation'])==64,'Fixed 64-window evaluation changed')
        frozen={name:value.detach().cpu().clone() for name,value in trainers['baseline'].model.state_dict().items()
                if name.startswith(('lam.','dynamics.'))}
        report['fresh_initialization_exact']=True
        save_common('initial.pt');save_common('last-common.pt')
        for pair in range(1,target+1):
            if not budget.permits(max(report.get('recent_pair_seconds',1)*2,30)):
                raise TimeoutError('Combined comparison allocation/export reserve reached')
            results=paired_update(trainers,budget,on_success,on_partial)
            report['latest_common_step']=pair;report['recent_pair_seconds']=sum(item['seconds'] for item in results.values())
            with (out/'updates.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=pair,arms=results))+'\n')
            if pair==1:save_common('measurement-boundary.pt')
            if pair==2:
                # Both finite updates count toward the unique trajectory. Only
                # the repeated second update is additional measurement compute.
                for arm in ARMS:
                    if not budget.permits(max(results[arm]['seconds']*2,30)):
                        raise TimeoutError('Budget ended before both resume measurements')
                    expected=cpu_copy(trainers[arm].snapshot())
                    replay=ComparisonTrainer(root,specifications[arm],args.experiment,arm,args.device,args.preflight,
                                             resume=out/arm/'measurement-boundary.pt')
                    started=time.monotonic();loss=replay.update(budget)
                    if loss is None:raise TimeoutError('Budget ended during exact-resume measurement')
                    report['optimizer_calls'][arm]+=1;report['replay_optimizer_calls'][arm]+=1
                    compare_exact(replay.snapshot(),expected)
                    check(loss==results[arm]['loss'],'Exact-resume loss differs')
                    report['measurement'][arm]=dict(status='passed',resume_kind='exact',maximum_state_difference=0,
                        step=2,loss=loss,update_seconds=results[arm]['seconds'],replay_seconds=time.monotonic()-started)
                    trainers[arm]=replay
                report['measurement']['both_arms_passed']=True
                save_common('measurement-checked.pt');save_common('last-common.pt');write_report()
            if pair in experiment['save_steps']:
                save_common(f'step-{pair:06d}.pt');save_common('last-common.pt')
                report['saved_common_steps'].append(pair)
                evaluate_common(pair)
            if pair%25==0 or pair==target:
                print(json.dumps(dict(step=pair,losses={arm:r['loss'] for arm,r in results.items()},remaining_seconds=budget.remaining())),flush=True)
                write_report()
        report['status']='completed'
    except TimeoutError as error:
        report.update(status='budget_exhausted',error=str(error))
    except BaseException as error:
        report.update(status='failed',exception=type(error).__name__,error=str(error))
        report['failures']={arm:trainer.last_failure for arm,trainer in trainers.items() if trainer.last_failure}
        raise
    finally:
        if frozen is not None:
            report['frozen_modules']={}
            for arm,trainer in trainers.items():
                try:
                    actual={name:value for name,value in trainer.model.state_dict().items() if name in frozen}
                    compare_exact(actual,frozen)
                    report['frozen_modules'][arm]=dict(unchanged=True,tensors=len(frozen),elements=sum(v.numel() for v in frozen.values()))
                except Exception as error:
                    report['frozen_modules'][arm]=dict(unchanged=False,error=str(error));report['status']='failed'
        if len(trainers)==2 and trainers['baseline'].step==trainers['detail'].step:
            report['latest_common_step']=trainers['baseline'].step
            try:save_common('last-common.pt')
            except Exception as error:report.update(status='failed',checkpoint_save_error=str(error))
        for arm,trainer in trainers.items():
            try:trainer.save(out/arm/'last.pt',budget)
            except Exception as error:report.setdefault('checkpoint_save_errors',{})[arm]=str(error);report['status']='failed'
        if torch.device(args.device).type=='cuda':report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['unique_updates_per_arm']={arm:trainer.step for arm,trainer in trainers.items()}
        write_report()
    check(report['status']!='failed','Comparison checkpoint save failed')
    print(json.dumps(dict(status=report['status'],latest_common_step=report['latest_common_step'],optimizer_calls=report['optimizer_calls'])),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    parser.add_argument('--baseline-spec',required=True);parser.add_argument('--detail-spec',required=True)
    parser.add_argument('--experiment',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--authorized',action='store_true');parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--updates',type=int);parser.add_argument('--rate',type=float,default=0)
    parser.add_argument('--balance',type=float);parser.add_argument('--allocation-start',type=float)
    run(parser.parse_args())


if __name__=='__main__':main()
