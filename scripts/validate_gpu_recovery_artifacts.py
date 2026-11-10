"""Validate downloaded GPU recovery evidence locally; never promote its weights."""
import argparse
import json
import math
from pathlib import Path

import torch

from genie_diep.cohort import read
from genie_diep.data import file_hash
from scripts.gpu_recovery_verification import compare_states


def require(condition,message):
    if not condition:raise ValueError(message)


def finite_tensors(value,label='state'):
    """Check every tensor, including optimizer moments, without loading onto CUDA."""
    if isinstance(value,torch.Tensor):
        require(bool(torch.isfinite(value).all()),f'Nonfinite tensor: {label}')
        return 1
    if isinstance(value,dict):return sum(finite_tensors(v,f'{label}.{k}') for k,v in value.items())
    if isinstance(value,(tuple,list)):return sum(finite_tensors(v,f'{label}[{i}]') for i,v in enumerate(value))
    if isinstance(value,float):require(math.isfinite(value),f'Nonfinite scalar: {label}')
    return 0


def validate_report(report,manifest):
    require(report['status']=='passed','Verification did not pass')
    require(report['preflight'] is True and report['promotion_approved'] is False and report['quality_gates']=='unmeasured',
            'Diagnostic-only status changed')
    for name in ('checkpoint','implementation','spec','model'):
        require(report[name+'_sha256']==manifest['expected_'+name+'_sha256'],f'{name} identity differs from exported overlay')
    members=manifest['member_hashes']
    require(report['harness_sha256']==members['scripts/gpu_recovery_verification.py'],'Harness identity changed')
    require(report['source_step']==219 and report['final_step']==518,'Unexpected source/final step')
    require(report['maximum_optimizer_calls']==300 and report['optimizer_calls']==300,'Optimizer-call accounting changed')
    require(report['maximum_compute_units']==2 and report['maximum_allocated_hours']==1 and report['minimum_remaining_units']==50,
            'Authorization limits changed')
    for key,cap in [('spent_seconds',3600),('estimated_units',2)]:
        require(math.isfinite(report[key]) and 0<=report[key]<=cap,f'Harness {key} exceeds budget')
    sustained=report['sustained']
    require(sustained['unique_additional_steps']==299 and sustained['final_step']==518 and sustained['measured_updates']==298,
            'Sustained-update accounting changed')
    recovery=report['recovery'];difference=recovery['max_absolute_state_difference']
    require(recovery['status']=='passed' and recovery['rollback_exact'] is True,'Failure recovery did not pass')
    require(math.isfinite(difference) and difference>=0,'Invalid replay difference')
    loss_a,loss_b=recovery['reference_loss'],recovery['replayed_loss']
    require(math.isfinite(loss_a) and math.isfinite(loss_b) and math.isclose(loss_a,loss_b,rel_tol=1e-4,abs_tol=1e-6),
            'Replay loss comparison failed')
    expected_kind='exact' if difference==0 and loss_a==loss_b else 'within FP16 tolerance'
    require(recovery['kind']==expected_kind,'Replay precision claim disagrees with measured difference')
    evaluation=report['evaluation']
    require(evaluation['complete'] is True and evaluation['samples']==64 and evaluation['sources']==16 and evaluation['step']==500,
            'Fixed held-out evaluation incomplete or wrong boundary')
    require(evaluation['stage']=='tokenizer' and evaluation['preflight'] is True and evaluation['quality_gate']=='unmeasured',
            'Evaluation promotion or stage changed')
    require(len(evaluation['by_source'])==16,'Evaluation source accounting changed')
    for metric in [evaluation['pooled'],evaluation['equal_source'],*evaluation['by_source'].values()]:
        require(math.isfinite(metric) and metric>=0,'Nonfinite held-out metric')
    precision=report['precision']
    require(precision['status']=='passed' and precision['complete'] is True and precision['optimizer_updates']==0,
            'Recovered-batch comparison incomplete')
    require(precision['checkpoint_file_unchanged'] is True and precision['failed_samples_recovered_exactly'] is True,
            'Source checkpoint or failed samples not verified')
    require(precision['checkpoint_sha256']==report['checkpoint_sha256'],'Precision source checkpoint differs')
    for name,member in [('diagnostic_source_sha256','genie_diep/gpu_precision.py'),
                        ('model_source_sha256','genie_diep/model.py'),
                        ('recovery_helper_sha256','scripts/diagnose_tokenizer_precision.py'),
                        ('specification_sha256','recovery-inputs/original-spec.json')]:
        require(precision[name]==members[member],f'Precision {name} changed')
    cases=precision['cases']
    require(len(cases)==3,'Missing precision comparisons')
    for case,(dtype,scale) in zip(cases,[('fp32',1),('fp16',65536),('fp16',1024)]):
        require(case['precision']==dtype and case['scale']==scale,'Precision cases changed')
        require(case['model_state_unchanged'] is True and case['optimizer_updates']==0,'Precision case mutated weights')
        require(case['status'] in ('finite','nonfinite'),'Precision case did not finish')
        if scale!=65536:
            require(case['status']=='finite' and len(case['microbatches'])==4,'Required conservative comparison failed')
        if case['status']=='finite':
            require(len(case['microbatches'])==4,'Finite case did not process the complete batch')
            for microbatch in case['microbatches']:
                require(microbatch['finite_loss'] and math.isfinite(microbatch['loss']) and
                        not microbatch['nonfinite_gradient_parameters'],'Finite case contains failed microbatch')
                require(all(g['nonfinite']==0 for g in microbatch['gradients']),'Finite case contains nonfinite gradients')
        else:
            require(any(not microbatch['finite_loss'] or any(g['nonfinite'] for g in microbatch.get('gradients',[]))
                        for microbatch in case['microbatches']),'Nonfinite case lacks nonfinite evidence')
    require(precision['fp32_finite'] is True and precision['conservative_fp16_finite'] is True,'Required precision flags failed')
    migration=report['migration']
    require(migration['kind']=='explicit_diagnostic_fork' and migration['preflight'] is True and migration['promotion_approved'] is False,
            'Migration was presented as production resume')
    require(migration['source_step']==219 and migration['failed_update']==220 and migration['source_sampler_draws']==3521 and
            migration['recovered_sampler_draws']==3505,'Failure sampling boundary changed')
    require(len(migration['recovered_windows'])==16 and migration['recovered_windows']==precision['samples'],
            'Recovered windows differ between precision and migration')
    require(migration['initial_scale']==1024 and migration['maximum_scale']==1024 and migration['scaler_reset'] is True,
            'Conservative scaler policy changed')
    failure=report['injected_failure']
    require(failure['restored_pre_update_boundary'] is True and failure['attempted_step']==220 and
            failure['phase']=='gradient_check' and failure['exception']=='FloatingPointError','Injected failure not safely recovered')
    require(failure['sampler_draws_before']==3505 and failure['sampler_draws_at_failure']==3521 and
            failure['sampled_windows']==migration['recovered_windows'],'Injected failure consumed different samples')


def validate(directory,overlay_manifest,source_checkpoint):
    directory=Path(directory).resolve();manifest=read(overlay_manifest)
    source_checkpoint=Path(source_checkpoint).resolve()
    require(file_hash(source_checkpoint)==manifest['expected_checkpoint_sha256'],'Preserved source checkpoint hash changed')
    report=read(directory/'verification-report.json');validate_report(report,manifest)
    require(read(directory/'precision-report.json')==report['precision'],'Separate precision report differs')
    updates=[json.loads(line) for line in (directory/'updates.jsonl').read_text(encoding='utf-8').splitlines() if line]
    require([item['step'] for item in updates]==list(range(221,519)),'Sustained update log has missing/duplicate steps')
    for item in updates:
        require(math.isfinite(item['loss']) and math.isfinite(item['seconds']) and item['seconds']>=0 and item['scale']==1024,
                'Update log contains invalid loss/timing/scale')
    source=torch.load(source_checkpoint,map_location='cpu',weights_only=True)
    require(source['step']==219 and source['contract']['preflight'] is False,'Preserved checkpoint is not the production failure')
    tensor_count=finite_tensors(source['model'],'source.model')+finite_tensors(source['optimizer'],'source.optimizer')
    required={'migration-boundary.pt':219,'injected-failure-boundary.pt':219,'last.pt':518}
    checkpoint_paths=sorted(directory.rglob('*.pt'))
    require(set(required)<=set(p.name for p in checkpoint_paths),'Missing required checkpoint')
    states={};checkpoint_records=[]
    for path in checkpoint_paths:
        state=torch.load(path,map_location='cpu',weights_only=True);contract=state['contract']
        require(contract['preflight'] is True and contract['stage']=='tokenizer' and contract['precision']=='fp16' and
                contract['device_type']=='cuda','Checkpoint is not a CUDA tokenizer diagnostic')
        for name in ('implementation','spec'):
            require(contract[name+'_sha256']==manifest['expected_'+name+'_sha256'],'Checkpoint source/configuration changed')
        require(contract==report['migration']['new_contract'],'Checkpoint contract differs from migrated branch')
        for key in ('binding_sha256','evaluation_sha256'):
            require(contract[key]==source['contract'][key],'Frozen data/evaluation binding changed')
        require(state['config']==source['config'],'Checkpoint architecture changed')
        require(contract['microbatch']==4 and contract['accumulation']==4,'Checkpoint batch size changed')
        require(state['sampler']['draws']==1+state['step']*16,'Checkpoint sampler is not a completed update boundary')
        require(state['sampler']['fraction']==source['sampler']['fraction'],'Sampling mixture changed')
        require(state['scaler']['scale']==1024,'Checkpoint scaler escaped cap')
        require(state['scheduler']['last_epoch']==state['step'],'Checkpoint scheduler/step mismatch')
        for slot in state['optimizer']['state'].values():
            require(float(slot['step'])==state['step'],'Adam step does not match completed update')
        count=finite_tensors(state['model'],path.name+'.model')+finite_tensors(state['optimizer'],path.name+'.optimizer')
        tensor_count+=count
        if path.name in required:
            require(path.parent==directory and state['step']==required[path.name],'Checkpoint stored at wrong boundary/path')
            states[path.name]=state
        checkpoint_records.append(dict(file=path.relative_to(directory).as_posix(),sha256=file_hash(path),step=state['step'],
                                       checked_tensors=count,preflight=True))
    boundary=states['migration-boundary.pt'];failed=states['injected-failure-boundary.pt'];last=states['last.pt']
    for key in ('model','optimizer','scheduler','rng'):
        compare_states(boundary[key],source[key],rtol=0,atol=0)
    for key in ('model','optimizer','scheduler','scaler','sampler','rng','step','contract'):
        compare_states(boundary[key],failed[key],rtol=0,atol=0)
    require(failed['last_failure']==report['injected_failure'],'Failure checkpoint diagnostic differs')
    require(last['last_failure'] is None,'Final checkpoint retains an unresolved failure')
    require(report['migration']['source_contract']==source['contract'],'Migration source contract differs')
    files=[dict(file=p.relative_to(directory).as_posix(),bytes=p.stat().st_size,sha256=file_hash(p))
           for p in sorted(directory.rglob('*')) if p.is_file()]
    return dict(schema_version=1,status='passed',artifact_directory=str(directory),
        overlay_manifest_sha256=file_hash(overlay_manifest),source_checkpoint_sha256=file_hash(source_checkpoint),
        checked_tensors=tensor_count,checkpoints=checkpoint_records,files=files,
        final_step=518,optimizer_calls=300,unique_additional_steps=299,evaluation_step=500,evaluation_windows=64,
        evaluation_sources=16,recovery_kind=report['recovery']['kind'],preflight=True,promotion_approved=False,
        quality_gates='unmeasured',allocation_accounting_scope='Harness estimates only; live allocation receipt must cover export and idle time separately.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True);parser.add_argument('--overlay-manifest',required=True)
    parser.add_argument('--source-checkpoint',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve previous validation; use a new output file')
    report=validate(args.directory,args.overlay_manifest,args.source_checkpoint)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('status','final_step','checked_tensors','recovery_kind')}),flush=True)


if __name__=='__main__':main()
