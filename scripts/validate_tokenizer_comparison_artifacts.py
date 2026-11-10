"""Read-only CPU verification of a downloaded paired tokenizer experiment.

Prints a receipt; does not change artifacts, restore training, decode test footage,
or independently repeat the GPU numerical-resume experiment.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from genie_diep.cohort import Cohort, MixedSampler, read
from genie_diep.data import file_hash
from scripts.run_tokenizer_comparison import ARMS, compare_exact, validate_experiment
from scripts.validate_gpu_recovery_artifacts import finite_tensors


def require(condition,message):
    if not condition:raise ValueError(message)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def load(path):return torch.load(path,map_location='cpu',weights_only=True)


def validate(root,directory,experiment_path):
    root=Path(root).resolve();directory=Path(directory).resolve();experiment=read(root/experiment_path)
    experiment=validate_experiment(root,experiment_path,*(experiment['arms'][arm]['spec'] for arm in ARMS))
    experiment_sha=file_hash(root/experiment_path);report=read(directory/'comparison-report.json')
    require(report['status'] in ('completed','budget_exhausted'),'No completed or bounded-stop comparison report')
    require(report['preflight'] is False and report['promotion_approved'] is False and report['quality_gates']=='unmeasured',
            'Production research / no-promotion status changed')
    require(report['device'].startswith('cuda'),'Comparison did not execute on CUDA')
    require(report['experiment_sha256']==experiment_sha and report['source_hashes']==experiment['source_hashes'],
            'Report does not match the immutable experiment')
    require(report['limits']==experiment['limits'],'Reported allocation limits changed')
    require(report['fresh_initialization_exact'] is True,'Matching fresh initialization not verified')
    common=report['latest_common_step']
    require(type(common) is int and 2<=common<=999,'No verified common comparison boundary')
    require(report['requested_unique_updates_per_arm']==999,'Requested unique-update cap changed')
    if report['status']=='completed':require(common==999,'Completed report stopped before requested boundary')
    require(report['measurement'].get('both_arms_passed') is True,'Both GPU measurements did not finish')
    require(not report.get('checkpoint_save_error') and not report.get('checkpoint_save_errors'),'Checkpoint saving failed')
    for field,maximum in [('spent_seconds',3600),('estimated_units',2)]:
        require(math.isfinite(report[field]) and 0<=report[field]<=maximum,f'Harness {field} exceeds authorization')
    saved=[step for step in experiment['save_steps'] if step<=common]
    require(report['saved_common_steps']==saved,'Saved comparison boundaries are missing or out of order')
    updates=jsonl(directory/'updates.jsonl')
    require([item['step'] for item in updates]==list(range(1,common+1)),'Paired update log has missing/duplicate steps')
    for item in updates:
        require(set(item['arms'])==set(ARMS),'Update log is missing an arm')
        for arm,value in item['arms'].items():
            require(value['step']==item['step'] and math.isfinite(value['loss']) and
                    math.isfinite(value['seconds']) and value['seconds']>=0 and value['scale']==1024,
                    'Update log has invalid step/loss/time/scale')
    partials=report.get('partial_checkpoints',[])
    require(len(partials)<=1,'More than one unmatched partial update was retained')
    for partial in partials:
        require(partial['arm'] in ARMS and partial['step']==common+1 and partial['comparable'] is False,
                'Partial checkpoint is not marked as an unmatched next update')
        require(partial['path']==f"{partial['arm']}/partial-step-{partial['step']}.pt",'Unexpected partial artifact path')
    specs={arm:read(root/experiment['arms'][arm]['spec']) for arm in ARMS}
    baseline_spec=specs['baseline']
    fixed=read(root/baseline_spec['evaluation_selection'])
    fixed_sources={window['source_id'] for window in fixed['validation']}
    require(len(fixed['validation'])==64 and len(fixed_sources)==16,'Frozen evaluation selection changed')
    for arm in ARMS:
        calls=report['optimizer_calls'][arm];replays=report['replay_optimizer_calls'][arm]
        discarded=sum(p['arm']==arm for p in partials)
        require(replays==1 and calls==common+replays+discarded and calls<=1000,'Optimizer/replay/partial accounting inconsistent')
        require(report['unique_updates_per_arm'][arm]==common,'Arm unique steps do not match the common boundary')
        measurement=report['measurement'][arm]
        require(measurement['status']=='passed' and measurement['resume_kind']=='exact' and measurement['maximum_state_difference']==0
                and measurement['step']==2,'GPU exact-resume measurement did not pass')
        require(measurement['loss']==updates[1]['arms'][arm]['loss'],'Resume loss disagrees with recorded second update')
        require(report['frozen_modules'][arm]['unchanged'] is True,'Frozen-module mutation was reported')
        evaluations=jsonl(directory/arm/'evaluation.jsonl') if saved else []
        recorded=[item['report'] for item in report.get('evaluations',[]) if item['arm']==arm]
        require(evaluations==recorded and [item['step'] for item in evaluations]==saved,'Evaluation files/report/boundaries differ')
        for evaluation in evaluations:
            require(evaluation['preflight'] is False and evaluation['quality_gate']=='unmeasured' and evaluation['stage']=='tokenizer',
                    'Evaluation eligibility changed')
            require(0<=evaluation['samples']<=64 and evaluation['sources']==len(evaluation['by_source']) and
                    set(evaluation['by_source'])<=fixed_sources,'Evaluation source/sample accounting invalid')
            require(evaluation['complete']==(evaluation['samples']==64),'Evaluation completion flag disagrees with count')
            if evaluation['complete']:require(evaluation['sources']==16,'Complete evaluation omitted a held-out source')
            if report['status']=='completed':require(evaluation['complete'],'Completed run contains an interrupted validation')
            for value in [evaluation['pooled'],evaluation['equal_source'],*evaluation['by_source'].values()]:
                require((value is None and evaluation['samples']==0) or
                        (value is not None and math.isfinite(value) and value>=0),'Invalid evaluation diagnostic')
    # Reproduce sampling state from metadata only: no image decoding is involved.
    data=Cohort(root,baseline_spec['data_binding'],'train',baseline_spec['model']['context'])
    sampler=MixedSampler(data,baseline_spec['optimization']['seed'],baseline_spec['data']['source_balanced_sampling_fraction'])
    steps={0,1,2,common,*saved,*(p['step'] for p in partials)};samplers={}
    for step in sorted(steps):
        while sampler.draws<1+step*16:sampler.draw()
        samplers[step]=sampler.state_dict()
    initials={arm:load(directory/arm/'initial.pt') for arm in ARMS}
    for key in ('model','optimizer','scheduler','scaler','sampler','rng','step','config'):
        compare_exact(initials['baseline'][key],initials['detail'][key])
    frozen={name:value for name,value in initials['baseline']['model'].items() if name.startswith(('lam.','dynamics.'))}
    checkpoints=[];checked_tensors=0;finals={}
    for arm in ARMS:
        expected_names={'initial.pt':0,'measurement-boundary.pt':1,'measurement-checked.pt':2,'last-common.pt':common,'last.pt':common}
        expected_names.update({f'step-{step:06d}.pt':step for step in saved})
        expected_names.update({f'partial-step-{p["step"]}.pt':p['step'] for p in partials if p['arm']==arm})
        paths=sorted((directory/arm).rglob('*.pt'))
        require({p.name for p in paths}==set(expected_names),'Unexpected/missing checkpoint files')
        for path in paths:
            require(path.parent==directory/arm,'Unexpected checkpoint subdirectory')
            state=load(path);contract=state['contract'];step=state['step']
            require(step==expected_names[path.name],'Checkpoint stored at the wrong boundary')
            require(contract['preflight'] is False and contract['stage']=='tokenizer' and contract['precision']=='fp16'
                    and contract['device_type']=='cuda','Checkpoint training contract changed')
            require(contract['torch_version']==report['torch_version'],'Checkpoint/runtime versions differ')
            require(contract['spec_sha256']==experiment['arms'][arm]['spec_sha256'] and
                    contract['implementation_sha256']==experiment['source_hashes']['implementation_sha256'],'Checkpoint source/spec binding changed')
            require(contract['comparison']==dict(experiment_sha256=experiment_sha,arm=arm,objective=experiment['arms'][arm]['objective'],
                    source_hashes=experiment['source_hashes']),'Checkpoint comparison/objective binding changed')
            require(contract['binding_sha256']==data.binding_sha256 and contract['evaluation_sha256']==file_hash(root/baseline_spec['evaluation_selection']),
                    'Checkpoint frozen data/evaluation binding changed')
            require(contract['microbatch']==4 and contract['accumulation']==4 and state['config']==baseline_spec['model'],'Model or effective batch changed')
            require(state['scheduler']['last_epoch']==step and state['scaler']['scale']==1024,'Scheduler or scaler boundary changed')
            require(state.get('last_failure') is None,'Saved comparison checkpoint retains a failure')
            compare_exact(state['sampler'],samplers[step])
            compare_exact({name:state['model'][name] for name in frozen},frozen)
            for slot in state['optimizer']['state'].values():require(float(slot['step'])==step,'Adam step disagrees with checkpoint step')
            if step==0:require(not state['optimizer']['state'],'Fresh checkpoint already has optimizer moments')
            count=finite_tensors(state['model'],arm+'.model')+finite_tensors(state['optimizer'],arm+'.optimizer');checked_tensors+=count
            checkpoints.append(dict(path=path.relative_to(directory).as_posix(),step=step,sha256=file_hash(path),finite_tensors=count))
            if path.name=='last-common.pt':finals[arm]=state
            if path.name=='last.pt':
                common_state=load(directory/arm/'last-common.pt')
                for key in ('model','optimizer','scheduler','scaler','sampler','rng','step','contract'):compare_exact(state[key],common_state[key])
        require(report['frozen_modules'][arm]['tensors']==len(frozen) and report['frozen_modules'][arm]['elements']==sum(v.numel() for v in frozen.values()),
                'Frozen-module report counts changed')
    compare_exact(finals['baseline']['sampler'],finals['detail']['sampler'])
    files=[dict(path=p.relative_to(directory).as_posix(),bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(directory.rglob('*')) if p.is_file()]
    return dict(schema_version=1,status='validated',run_status=report['status'],latest_common_step=common,
        optimizer_calls=report['optimizer_calls'],experiment_sha256=experiment_sha,finite_tensors_checked=checked_tensors,
        initial_model_and_rng_match=True,sampling_independently_replayed=True,frozen_lam_dynamics_exact=True,
        gpu_resume_evidence='Both frozen GPU runner measurements report exact equality; this CPU check validates saved evidence and does not re-execute CUDA.',
        accounting_limit='Harness totals are shared; separately verify original allocation start, live charges, export/idle and runtime release.',
        quality_gates='unmeasured',promotion_approved=False,checkpoints=checkpoints,files=files)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.');parser.add_argument('--directory',required=True)
    parser.add_argument('--experiment',default='configs/tokenizer-comparison-v1.json')
    args=parser.parse_args();torch.set_num_threads(2)
    print(json.dumps(validate(args.root,args.directory,args.experiment),indent=2),flush=True)


if __name__=='__main__':main()
