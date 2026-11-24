"""Prospective cap12 extension of the proven transactional FP32 policy.

Only the hash-bound safe3453 lineage can migrate once to a fresh branch.
Migration changes contract metadata only, preserves every event, and adapts the
original initial identically. No new training method, optimizer or data order.
"""
import argparse
import copy
from contextlib import contextmanager
from pathlib import Path
import shutil

import torch
from genie_diep.cohort import Cohort, read
from genie_diep.data import file_hash
from scripts import recover_staged_tokenizer_fp32 as proven
from scripts import run_staged_tokenizer as original
from scripts.run_tokenizer_comparison import source_hash, compare_exact

POLICY=dict(proven.POLICY,maximum_joint_events=12)
VERSION='staged16-fp32-cap12-extension-v1'
require,write_json=proven.require,proven.write_json


@contextmanager
def policy_scope():
    # Reuse the exact successful transaction implementation; only its review
    # allowance differs. Nested calls restore the prior scope deterministically.
    previous=proven.POLICY;proven.POLICY=POLICY
    try:yield
    finally:proven.POLICY=previous


def validate_manifest(root,path):
    manifest=read(path)
    require(manifest['schema_version']==1 and manifest['version']==VERSION and manifest['policy']==POLICY,'Extension policy changed')
    require(manifest['source_sha256']==source_hash(__file__) and
        manifest['proven_source_sha256']==source_hash(proven.__file__),'Extension/proven source changed')
    require(manifest['support_source_sha256']==source_hash(proven.prior.__file__),'Support source changed')
    require(manifest['accepted_step']==3453 and manifest['inherited_event_updates']==[1804,1805,2902,3453],
        'Extension is restricted to the reviewed safe3453 lineage')
    require(manifest['old_maximum_joint_events']==4 and manifest['new_maximum_joint_events']==12,'Cap cannot reset or change')
    require(manifest['original_plan_sha256']==file_hash(Path(root)/manifest['plan']),'Original plan changed')
    original.validate_plan(root,manifest['plan'])
    return manifest


def extension_contract(contract,manifest_path,manifest):
    result=copy.deepcopy(contract)
    require('policy_extension' not in result,'A migrated lineage cannot migrate again')
    require(result['precision_exception']==dict(manifest_sha256=manifest['historical_manifest_sha256'],
        source_sha256=manifest['proven_source_sha256'],policy=proven.POLICY),'Not the proven original FP32 contract')
    result['precision_exception']=dict(manifest_sha256=file_hash(manifest_path),source_sha256=source_hash(__file__),policy=POLICY)
    result['policy_extension']=dict(version=VERSION,parent_checkpoint_sha256=manifest['parent_checkpoint_sha256'],
        historical_manifest_sha256=manifest['historical_manifest_sha256'],historical_proof_sha256=manifest['historical_proof_sha256'],
        inherited_events=4,old_maximum=4,new_maximum=12)
    return result


def adapt_state(state,manifest_path,manifest):
    result=copy.deepcopy(state);result['contract']=extension_contract(state['contract'],manifest_path,manifest)
    for key in state:
        if key!='contract':compare_exact(result[key],state[key])
    return result


def validate_source(root,manifest_path,source_run,history_run):
    manifest=validate_manifest(root,manifest_path);source=Path(source_run).resolve();history=Path(history_run).resolve()
    files=[(source/'experiment-report.json','parent_report_sha256'),(source/'joint/step-003453.pt','parent_checkpoint_sha256'),
        (source/'fp32-exception-events.json','parent_events_sha256'),(history/'fp32-verification.json','historical_proof_sha256'),
        (history/'verified-step-001809.pt','historical_checkpoint_sha256'),(history/'joint/initial.pt','parent_initial_sha256')]
    for path,key in files:require(file_hash(path)==manifest[key],'Bound extension input changed: '+path.name)
    report=read(files[0][0]);journal=read(files[2][0]);proof=read(files[3][0])
    require(report['status']=='stopped' and report['error']['message']=='FP32 event cap reached; safe review required',
        'Parent was not stopped solely by the reviewed governor')
    identity=report['identity'];require(identity['mode']=='full' and identity['local_check'] is False and
        identity['plan_sha256']==manifest['original_plan_sha256'] and identity['budget_sha256']==manifest['original_budget_sha256'],
        'Parent experiment identity changed')
    require(proof['status']=='passed' and proof['source_sha256']==manifest['proven_source_sha256'] and
        proof['manifest_sha256']==manifest['historical_manifest_sha256'] and proof['reference']['exact'] and
        proof['replay']['passed'] and proof['replay']['maximum_state_difference']==0,'Historical CUDA proof changed')
    state=torch.load(files[1][0],map_location='cpu',weights_only=True)
    old=torch.load(files[4][0],map_location='cpu',weights_only=True)
    initial=torch.load(files[5][0],map_location='cpu',weights_only=True)
    require(state['step']==report['phases']['joint']['latest_step']==3453 and
        report['phases']['joint']['checkpoint']['sha256']==manifest['parent_checkpoint_sha256'],'Parent step changed')
    require(state['contract']==old['contract']==initial['contract'],'Parent/initial/verified contracts differ')
    extension_contract(state['contract'],manifest_path,manifest)
    events=state['fp32_exception_events'];require([e['attempted_update'] for e in events]==manifest['inherited_event_updates'],
        'Inherited event history changed')
    compare_exact(events,journal['events']);require(journal['pending'] is None,'Unresolved event journal')
    compare_exact(events[:2],proof['fp32_exception_events'])
    for event in events:
        require(event['status']=='accepted' and event['precision']=='fp32' and event['dropped_samples']==0 and
            event['accepted_step_before']+1==event['accepted_step_after']==event['attempted_update'] and
            len(event['sampled_windows'])==16,'Invalid inherited accepted event')
        if event['attempted_update']==1804:continue
        reason=event['reason'];failure=reason['failure']
        require(reason['kind']=='recurrent_AMP_gradient_overflow' and reason['exact_rollback'] and
            failure['phase']=='gradient_check' and failure['message']=='Nonfinite gradient; no optimizer update' and
            failure['restored_pre_update_boundary'] and failure['scale']==1024 and failure['completed_microbatches']==4,
            'Inherited exception was not eligible')
    proven.finite_tree(state['model']);proven.finite_tree(state['optimizer'])
    require({int(v['step']) for v in state['optimizer']['state'].values()}=={3453},'Adam update count changed')
    require(state['scheduler']['last_epoch']==3453 and state['scaler']['scale']==1024 and
        state['scaler']['_growth_tracker']==1449,'Schedule/AMP progression changed')
    for key in state['model']:
        if not key.startswith('tokenizer.'):compare_exact(state['model'][key],old['model'][key])
    plan=read(Path(root)/manifest['plan']);spec=read(Path(root)/plan['stages']['joint']['spec'])
    data=Cohort(root,spec['data_binding'],'train',16);sampler=proven.prior.recover_sampler(data,old,plan,spec)
    indexed={e['attempted_update']:e for e in events[2:]}
    for update in range(1810,3454):
        descriptors=[data.descriptor(*sampler.draw()) for _ in range(16)]
        if update in indexed:require(descriptors==indexed[update]['sampled_windows'],'Inherited event data order changed')
    compare_exact(sampler.state_dict(),state['sampler'])
    return manifest,report,state,initial,proof


def context_for(manifest_path,manifest,out):
    return dict(manifest_sha256=file_hash(manifest_path),failed_proof_sha256=manifest['failed_proof_sha256'],
        out=str(out),extension=manifest,extension_manifest_path=str(manifest_path))


class ExtensionTrainer(proven.ExceptionTrainer):
    def __init__(self,*args,context,resume=None,**kwargs):
        # original.run passes all nine parameters positionally; normalize the
        # resume/local_check/mode tail before passing an explicit empty resume.
        positional=list(args)
        if len(positional)>6:
            require(resume is None,'Ambiguous resume');resume=positional[6];positional[6]=None
            if len(positional)>7:
                require('local_check' not in kwargs,'Ambiguous local_check');kwargs['local_check']=positional[7]
            if len(positional)>8:
                require('mode' not in kwargs,'Ambiguous mode');kwargs['mode']=positional[8]
            positional=positional[:6]
        with policy_scope():super().__init__(*positional,context=context,resume=None,**kwargs)
        manifest=context['extension']
        self.contract['precision_exception']=dict(manifest_sha256=context['manifest_sha256'],source_sha256=source_hash(__file__),policy=POLICY)
        self.contract['policy_extension']=dict(version=VERSION,parent_checkpoint_sha256=manifest['parent_checkpoint_sha256'],
            historical_manifest_sha256=manifest['historical_manifest_sha256'],historical_proof_sha256=manifest['historical_proof_sha256'],
            inherited_events=4,old_maximum=4,new_maximum=12)
        if resume:
            state=torch.load(resume,map_location='cpu',weights_only=True)
            require(state['contract']==self.contract,'Exact cap12 resume contract changed')
            proven.load_boundary(self,state);self.fp32_events=copy.deepcopy(state['fp32_exception_events'])
            proven.anchor_inactive(self,state)

    def update(self,budget=None):
        with policy_scope():return super().update(budget)

    def fp32_update(self,budget,reason):
        with policy_scope():return super().fp32_update(budget,reason)


def migrate(root,manifest_path,source_run,history_run,out,budget=None):
    manifest,report,state,initial,proof=validate_source(root,manifest_path,source_run,history_run)
    out=Path(out).resolve();require(not out.exists(),'Fresh extension branch required; never reset an existing cap')
    adapted=adapt_state(state,manifest_path,manifest);initial_adapter=adapt_state(initial,manifest_path,manifest)
    require(adapted['contract']==initial_adapter['contract'],'Initial/endpoint adapter contracts differ')
    out.mkdir(parents=True);folder=out/'joint';folder.mkdir()
    checkpoint=folder/'step-003453.pt';torch.save(adapted,checkpoint);torch.save(initial_adapter,folder/'initial.pt')
    immutable=out/'migrated-step-003453.pt';shutil.copyfile(checkpoint,immutable)
    for src,name in [(Path(source_run)/'experiment-report.json','parent-experiment-report.json'),
        (Path(source_run)/'fp32-exception-events.json','parent-exception-events.json'),
        (Path(history_run)/'fp32-verification.json','historical-fp32-verification.json')]:shutil.copyfile(src,out/name)
    entry=report['phases']['joint'];entry['inherited_saved_checkpoints']=entry.get('saved_checkpoints',{})
    entry.update(checkpoint=dict(path='joint/step-003453.pt',sha256=file_hash(checkpoint)),
        saved_checkpoints={'3453':dict(path='joint/step-003453.pt',sha256=file_hash(checkpoint))},
        initial_checkpoint_sha256=file_hash(folder/'initial.pt'))
    report['status']='policy_extension_ready';report.pop('error',None)
    report['policy_extension']=dict(manifest_sha256=file_hash(manifest_path),source_sha256=source_hash(__file__),
        parent_checkpoint_sha256=manifest['parent_checkpoint_sha256'],historical_proof_sha256=manifest['historical_proof_sha256'],
        inherited_events=4,maximum_events=12)
    if budget:report['spent_seconds'],report['spent_units']=budget.spent()
    write_json(out/'experiment-report.json',report)
    receipt=dict(status='ready',manifest_sha256=file_hash(manifest_path),source_sha256=source_hash(__file__),
        parent_checkpoint_sha256=manifest['parent_checkpoint_sha256'],historical_proof_sha256=manifest['historical_proof_sha256'],
        migrated_checkpoint_sha256=file_hash(immutable),initial_adapter_sha256=file_hash(folder/'initial.pt'),
        numeric_state_unchanged=True,inherited_event_count=4,maximum_joint_events=12)
    write_json(out/'extension-receipt.json',receipt);return receipt


def validate_resume_report(out,manifest_path,manifest):
    out=Path(out).resolve();sha=file_hash(manifest_path);receipt=read(out/'extension-receipt.json')
    require(receipt['status']=='ready' and receipt['manifest_sha256']==sha and receipt['source_sha256']==source_hash(__file__),
        'Exact completed one-time migration required')
    require(file_hash(out/'migrated-step-003453.pt')==receipt['migrated_checkpoint_sha256'],'Immutable migrated anchor changed')
    require(file_hash(out/'historical-fp32-verification.json')==manifest['historical_proof_sha256'],'Historical proof changed')
    report=read(out/'experiment-report.json');identity=report['identity'];lineage=report['policy_extension']
    require(identity['mode']=='full' and identity['local_check'] is False and identity['plan_sha256']==manifest['original_plan_sha256'] and
        identity['budget_sha256']==manifest['original_budget_sha256'],'Progressive identity changed')
    require(lineage['manifest_sha256']==sha and lineage['source_sha256']==source_hash(__file__) and
        lineage['parent_checkpoint_sha256']==manifest['parent_checkpoint_sha256'],'Extension lineage changed')
    baseline=torch.load(out/'migrated-step-003453.pt',map_location='cpu',weights_only=True);steps={}
    for phase,entry in report['phases'].items():
        if 'checkpoint' not in entry:continue
        path=(out/entry['checkpoint']['path']).resolve()
        require(path.is_relative_to(out) and file_hash(path)==entry['checkpoint']['sha256'],'Progressive checkpoint changed')
        state=torch.load(path,map_location='cpu',weights_only=True)
        require(state['step']==entry['latest_step'] and state['contract']['precision_exception']==dict(
            manifest_sha256=sha,source_sha256=source_hash(__file__),policy=POLICY) and
            state['contract']['policy_extension']==baseline['contract']['policy_extension'],'Progressive cap12 contract changed')
        if phase=='joint':
            compare_exact(state['fp32_exception_events'][:4],baseline['fp32_exception_events'])
            require(4<=len(state['fp32_exception_events'])<=12 and (state['step']>=5000 or len(state['fp32_exception_events'])<12),
                'Cap12 exhausted or event history reset; review before allocation')
        steps[phase]=state['step']
    return steps


def run(args):
    original.configure_backend();torch.set_num_threads(args.threads)
    require(args.device=='cuda' and args.authorized and torch.cuda.is_available(),'Authorized CUDA only')
    manifest=validate_manifest(args.root,args.extension_manifest)
    require(args.plan==manifest['plan'] and file_hash(args.full_budget)==manifest['original_budget_sha256'],'Plan/budget changed')
    mode=args.mode;args.mode='full';args.local_check=False
    budget,_=original.allocation_budget(args);plan=original.validate_plan(args.root,args.plan)
    out=original.validate_output(args.root,args.out,args.device,args.private_output_root)
    if mode=='migrate':
        original.full_pilot_gate(args.root,args,plan,budget,{'joint':3453})
        return migrate(args.root,args.extension_manifest,args.source_run,args.history_run,out,budget)
    validate_resume_report(out,args.extension_manifest,manifest)
    context=context_for(args.extension_manifest,manifest,out);original_class=original.StagedTrainer
    original.StagedTrainer=lambda *a,**kw:ExtensionTrainer(*a,**kw,context=context);args.resume=True
    try:return original.run(args)
    finally:original.StagedTrainer=original_class;args.mode=mode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('migrate','resume'),required=True);p.add_argument('--root',required=True)
    p.add_argument('--plan',required=True);p.add_argument('--extension-manifest',required=True)
    p.add_argument('--source-run',required=True);p.add_argument('--history-run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--private-output-root');p.add_argument('--full-budget',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--authorized',action='store_true')
    p.add_argument('--rate',type=float,default=0);p.add_argument('--balance',type=float);p.add_argument('--allocation-start',type=float)
    p.add_argument('--prior-seconds',type=float,default=0);p.add_argument('--prior-units',type=float,default=0)
    p.add_argument('--threads',type=int,default=2);run(p.parse_args())

if __name__=='__main__':main()
