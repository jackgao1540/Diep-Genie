"""Bounded GPU tokenizer recovery verification; every checkpoint is diagnostic only."""
import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from genie_diep.baseline import Budget, Trainer, implementation_hash, restore_rng, stage_config, validate_readiness
from genie_diep.cohort import MixedSampler, read
from genie_diep.data import file_hash
from scripts.diagnose_tokenizer_precision import recover_consumed_batch


def verification_budget(rate,balance,allocation_start,now=None):
    now=time.time() if now is None else now
    if not all(math.isfinite(v) for v in (rate,balance,allocation_start,now)):
        raise ValueError('Allocation accounting must be finite')
    if rate<=0 or balance<=50 or allocation_start<=0 or allocation_start>now:
        raise ValueError('Live rate/balance and a past allocation start are required')
    return Budget(dict(max_allocated_hours=1,max_compute_units=2),rate=rate,balance=balance,
                  setup_seconds=now-allocation_start,reserve_seconds=180)


def migrate_legacy_failure(trainer,state,original_spec):
    """Explicit diagnostic fork, never weaken Trainer's exact-resume contract."""
    if not trainer.preflight or trainer.stage!='tokenizer':
        raise ValueError('Migration is restricted to a diagnostic tokenizer branch')
    old=state['contract']
    if old['stage']!='tokenizer' or state['config']!=trainer.model.config_dict():
        raise ValueError('Checkpoint stage or architecture changed')
    if old['binding_sha256']!=trainer.data.binding_sha256 or old['evaluation_sha256']!=trainer.contract['evaluation_sha256']:
        raise ValueError('Frozen data or evaluation changed')
    if (old['microbatch'],old['accumulation'])!=(trainer.microbatch,trainer.accumulation):
        raise ValueError('Failure batch configuration changed')
    for key in ('seed','weight_decay','gradient_clip_norm'):
        if original_spec['optimization'][key]!=trainer.spec['optimization'][key]:
            raise ValueError('Optimizer configuration changed outside precision policy')
    for key in ('learning_rate','max_updates','color_balance'):
        if stage_config(original_spec,'tokenizer')[key]!=trainer.stage_spec[key]:
            raise ValueError('Stage configuration changed outside precision policy')
    windows=recover_consumed_batch(trainer.data,state,original_spec['optimization']['seed'],
                                   original_spec['data']['source_balanced_sampling_fraction'])
    sampler=MixedSampler(trainer.data,original_spec['optimization']['seed'],
                         original_spec['data']['source_balanced_sampling_fraction'])
    before_draws=state['sampler']['draws']-len(windows)
    for _ in range(before_draws):sampler.draw()
    trainer.model.load_state_dict(state['model']);trainer.optimizer.load_state_dict(state['optimizer'])
    trainer.scheduler.load_state_dict(state['scheduler']);trainer.scaler=trainer._new_scaler()
    trainer.sampler.load_state_dict(sampler.state_dict());restore_rng(state['rng'])
    trainer.step=state['step'];trainer.best=state['best'];trainer.spent_seconds=0;trainer.spent_units=0
    trainer.last_failure=None;trainer.optimizer.zero_grad(set_to_none=True)
    return dict(kind='explicit_diagnostic_fork',source_step=state['step'],failed_update=state['step']+1,
        source_contract=old,new_contract=trainer.contract,source_sampler_draws=state['sampler']['draws'],
        recovered_sampler_draws=before_draws,recovered_windows=[trainer.data.descriptor(*w) for w in windows],
        model_optimizer_scheduler_restored=True,source_rng_restored=True,
        rng_basis='Tokenizer forward has no stochastic layers; source RNG retained while sampler is rewound by verified replay.',
        scaler_reset=True,initial_scale=trainer.scaler_initial_scale,maximum_scale=trainer.scaler_max_scale,
        preflight=True,promotion_approved=False)


def compare_states(actual,expected,rtol=1e-4,atol=1e-6):
    """Compare tensor/nested state and return the largest tensor difference."""
    if isinstance(expected,torch.Tensor):
        actual=actual.detach().cpu();expected=expected.detach().cpu()
        if not actual.is_floating_point():
            if not torch.equal(actual,expected):raise AssertionError('Discrete state changed')
            return 0.
        torch.testing.assert_close(actual,expected,rtol=rtol,atol=atol)
        return float((actual-expected).abs().max()) if actual.numel() else 0.
    if isinstance(expected,dict):
        if actual.keys()!=expected.keys():raise AssertionError('State keys changed')
        return max((compare_states(actual[k],v,rtol,atol) for k,v in expected.items()),default=0.)
    if isinstance(expected,(tuple,list)):
        if len(actual)!=len(expected):raise AssertionError('State length changed')
        return max((compare_states(a,b,rtol,atol) for a,b in zip(actual,expected)),default=0.)
    if actual!=expected:raise AssertionError('Scalar state changed')
    return 0.


def require_budget(budget,estimate=60):
    if not budget.permits(estimate):raise TimeoutError('Verification allocation/export reserve reached')


def model_source_hash(root):
    return hashlib.sha256((Path(root)/'genie_diep/model.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def run(args):
    if not 2<=args.updates<=300:raise ValueError('Verification permits 2 to 300 optimizer calls including replay')
    budget=verification_budget(args.rate,args.balance,args.allocation_start)
    root=Path(args.root).resolve();out=Path(args.out).resolve();checkpoint=Path(args.checkpoint).resolve()
    if out.exists():raise FileExistsError('Use a fresh verification output directory')
    expected=[(checkpoint,args.expected_checkpoint_sha256),(root/args.spec,args.expected_spec_sha256)]
    for path,sha in expected:
        if file_hash(path)!=sha:raise ValueError(f'Expected content changed: {path.name}')
    if implementation_hash()!=args.expected_implementation_sha256:raise ValueError('Implementation changed')
    if model_source_hash(root)!=args.expected_model_sha256:raise ValueError('Model source changed')
    validate_readiness(root,args.spec)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    original_spec=read(root/args.original_spec)
    if file_hash(root/args.original_spec)!=state['contract']['spec_sha256']:
        raise ValueError('Original specification does not match checkpoint')
    if state['step']!=219 or state['contract']['preflight']:
        raise ValueError('This verification is bound to the preserved production step-219 failure')
    if not torch.cuda.is_available():raise RuntimeError('Authorized GPU allocation required')
    torch.set_num_threads(args.threads);out.mkdir(parents=True)
    report=dict(schema_version=1,status='incomplete',preflight=True,promotion_approved=False,
        quality_gates='unmeasured',purpose='Precision, recovery and bounded tokenizer stability diagnostic only',
        implementation_sha256=implementation_hash(),spec_sha256=file_hash(root/args.spec),
        model_sha256=model_source_hash(root),checkpoint_sha256=file_hash(checkpoint),
        harness_sha256=file_hash(__file__),gpu=torch.cuda.get_device_name(),torch_version=str(torch.__version__),
        maximum_optimizer_calls=args.updates,optimizer_calls=0,source_step=219,maximum_allocated_hours=1,
        maximum_compute_units=2,minimum_remaining_units=50)
    trainer=None
    def save_report():
        report['spent_seconds'],report['estimated_units']=budget.spent()
        (out/'verification-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    def evaluate_fixed():
        require_budget(budget,120)
        if len(trainer.evaluation['validation'])!=64:raise ValueError('Fixed 64-window evaluation changed')
        report['evaluation']=trainer.evaluate(budget=budget)
        if not report['evaluation']['complete']:raise TimeoutError('Fixed evaluation interrupted by budget')
        save_report()
    try:
        from genie_diep.gpu_precision import compare_recovered_tokenizer_batch
        require_budget(budget,120)
        report['precision']=compare_recovered_tokenizer_batch(root,checkpoint,root/args.original_spec,
            output=out/'precision-report.json',budget=budget)
        if report['precision']['status']!='passed':raise FloatingPointError('Required FP32 / conservative FP16 diagnostic failed')
        require_budget(budget,120);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        trainer=Trainer(root,args.spec,'tokenizer','cuda','fp16',state['contract']['microbatch'],state['contract']['accumulation'],True)
        report['migration']=migrate_legacy_failure(trainer,state,original_spec)
        trainer.save(out/'migration-boundary.pt',budget)
        boundary=trainer._update_boundary()
        expected_loss=trainer.update(budget)
        if expected_loss is None:raise TimeoutError('Budget reached during reference update')
        report['optimizer_calls']+=1
        expected=trainer._update_boundary()
        del trainer;trainer=None;gc.collect();torch.cuda.empty_cache()
        require_budget(budget,120)
        trainer=Trainer(root,args.spec,'tokenizer','cuda','fp16',state['contract']['microbatch'],state['contract']['accumulation'],True,
                        resume=out/'migration-boundary.pt')
        handle=trainer.parameters[0].register_hook(lambda grad:grad*float('inf'))
        try:
            try:trainer.update(budget)
            except FloatingPointError:pass
            else:raise AssertionError('Injected nonfinite update did not fail')
        finally:handle.remove()
        if not trainer.last_failure or not trainer.last_failure['restored_pre_update_boundary']:
            raise AssertionError('Failed update did not restore its sampling boundary')
        compare_states(trainer._update_boundary(),boundary,rtol=0,atol=0)
        report['injected_failure']=copy.deepcopy(trainer.last_failure)
        trainer.save(out/'injected-failure-boundary.pt',budget)
        # Reload the actual failure checkpoint to verify persistent recovery, too.
        del trainer;trainer=None;gc.collect();torch.cuda.empty_cache()
        trainer=Trainer(root,args.spec,'tokenizer','cuda','fp16',state['contract']['microbatch'],state['contract']['accumulation'],True,
                        resume=out/'injected-failure-boundary.pt')
        repeated=trainer.update(budget)
        if repeated is None:raise TimeoutError('Budget reached during recovery replay')
        report['optimizer_calls']+=1
        difference=compare_states(trainer._update_boundary(),expected)
        if not math.isclose(repeated,expected_loss,rel_tol=1e-4,abs_tol=1e-6):raise AssertionError('Recovery loss differs')
        report['recovery']=dict(status='passed',max_absolute_state_difference=difference,
            kind='exact' if difference==0 and repeated==expected_loss else 'within FP16 tolerance',
            reference_loss=expected_loss,replayed_loss=repeated,rollback_exact=True)
        del boundary,expected;gc.collect();torch.cuda.empty_cache()
        times=[];last_save=time.monotonic()
        while report['optimizer_calls']<args.updates:
            require_budget(budget,max(times[-10:],default=1)*2+30)
            torch.cuda.synchronize();started=time.monotonic();loss=trainer.update(budget)
            if loss is None:raise TimeoutError('Budget reached during sustained verification')
            torch.cuda.synchronize();seconds=time.monotonic()-started;times.append(seconds)
            report['optimizer_calls']+=1
            item=dict(step=trainer.step,loss=loss,seconds=seconds,scale=trainer.scaler.get_scale())
            with (out/'updates.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(item)+'\n')
            if trainer.step%25==0:print(json.dumps(item),flush=True);save_report()
            if trainer.step==500:evaluate_fixed()
            if time.monotonic()-last_save>=60:trainer.save(out/'last.pt',budget);last_save=time.monotonic()
        report['sustained']=dict(unique_additional_steps=trainer.step-state['step'],final_step=trainer.step,
            measured_updates=len(times),mean_update_seconds=sum(times)/len(times) if times else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
        if 'evaluation' not in report:evaluate_fixed()
        if file_hash(checkpoint)!=report['checkpoint_sha256']:raise AssertionError('Preserved source checkpoint changed')
        report['status']='passed'
    except BaseException as error:
        report.update(status='incomplete' if isinstance(error,TimeoutError) else 'failed',
                      exception=type(error).__name__,error=str(error))
        if trainer and trainer.last_failure:report['failure']=trainer.last_failure
        raise
    finally:
        if trainer:
            report['final_step']=trainer.step
            try:trainer.save(out/'last.pt',budget)
            except Exception as error:report['checkpoint_save_error']=str(error);report['status']='failed'
        save_report()
    print(json.dumps(dict(status=report['status'],final_step=trainer.step,optimizer_calls=report['optimizer_calls'])),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.');parser.add_argument('--spec',default='configs/training-baseline-v1.json')
    for name in ('checkpoint','original-spec','out','expected-checkpoint-sha256','expected-implementation-sha256',
                 'expected-spec-sha256','expected-model-sha256'):parser.add_argument('--'+name,required=True)
    for name in ('rate','balance','allocation-start'):parser.add_argument('--'+name,type=float,required=True)
    parser.add_argument('--updates',type=int,default=300);parser.add_argument('--threads',type=int,default=2)
    run(parser.parse_args())


if __name__=='__main__':main()
