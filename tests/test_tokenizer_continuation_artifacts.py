import copy
import json
import math
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from genie_diep.baseline import rng_state
from genie_diep.cohort import MixedSampler
from genie_diep.data import file_hash
from scripts import validate_tokenizer_continuation_artifacts as audit
from scripts.run_tokenizer_continuation import EVALUATE_STEPS, LIMITS, SAVE_STEPS


def plan_template():
    return dict(parent_step=999, target_step=5000, limits=LIMITS, save_steps=SAVE_STEPS,
                evaluate_steps=EVALUATE_STEPS, sampling_policy='full_train_frame_coverage_then_parent_mixed',
                coverage_seed=20261110, source_hashes={'runner_sha256':'runner', 'coverage_sampler_sha256':'sampler'},
                parents={arm:{'sha256':arm, 'path':f'continuation-inputs/{arm}-step999.pt'} for arm in audit.ARMS})


def report_template(plan, common=5000):
    complete = common == 5000
    updates = [dict(step=step, arms={a:dict(step=step, scale=1024, loss=1., seconds=.5) for a in audit.ARMS})
               for step in range(1000, common+1)]
    measured = common >= 1001
    report = dict(status='completed' if complete else 'budget_exhausted', preflight=False, promotion_approved=False,
                  quality_gates='unmeasured',
                  device='cuda', experiment_sha256='experiment', limits=LIMITS, parent_step=999, target_step=5000,
                  sampling_change=plan['sampling_policy'], latest_common_step=common, spent_seconds=100.,
                  estimated_units=.1, peak_allocated_bytes=12345, torch_version='fixture-cuda',
                  optimizer_calls={a:common-999+int(measured) for a in audit.ARMS},
                  replay_optimizer_calls={a:int(measured) for a in audit.ARMS},
                  unique_new_updates_per_arm={a:common-999 for a in audit.ARMS},
                  migrations={}, measurement={}, frozen_modules={a:dict(unchanged=True,tensors=98) for a in audit.ARMS},
                  coverage={a:{} for a in audit.ARMS}, saved_common_steps=[s for s in SAVE_STEPS if s <= common], evaluations=[])
    for arm in audit.ARMS:
        report['migrations'][arm] = dict(model_exact=True, optimizer_exact=True, scheduler_exact=True, scaler_exact=True,
            rng_exact=True, parent_step=999, parent_sha256=plan['parents'][arm]['sha256'],
            only_training_change='shared complete-coverage sampling schedule', preflight=False)
        if measured:
            report['measurement'][arm] = dict(status='passed', maximum_state_difference=0, step=1001, seconds=.5, loss=1.)
    if measured:
        report['measurement']['both_arms_passed'] = True
    return report, updates


def selection():
    return dict(training_calibration=[{'source_id':f'train-{i}'} for i in range(84)],
                validation=[{'source_id':f'val-{i//4}'} for i in range(64)])


def assessment(split, step, count=None):
    rows = selection()['training_calibration' if split == 'train' else 'validation']
    n = len(rows) if count is None else count
    sources = audit.Counter(row['source_id'] for row in rows[:n])
    metrics = dict(mse=.1, foreground_mse=.2, temporal_mse=.3)
    return dict(split=split, step=step, samples=n, sources=len(sources), complete=n == len(rows),
                by_source={source:dict(samples=size, **metrics) for source,size in sources.items()},
                **(metrics if n else {key:None for key in metrics}))


def add_assessments(report):
    for step in EVALUATE_STEPS:
        if step <= report['latest_common_step']:
            for arm in audit.ARMS:
                for split in ('train', 'val'):
                    report['evaluations'].append(dict(arm=arm, common_step=step, report=assessment(split, step)))


def test_completed_accounting_counts_replay_inside_each_4002_call_cap():
    plan = plan_template(); report, updates = report_template(plan)
    common, saved, partial = audit.validate_report(report, plan, 'experiment', updates)
    assert common == 5000 and saved == SAVE_STEPS and partial == []


@pytest.mark.parametrize('mutate,match', [
    (lambda r:r.update(preflight=True), 'production'),
    (lambda r:r.update(promotion_approved=True), 'production'),
    (lambda r:r.update(quality_gates='pass'), 'production'),
    (lambda r:r.update(experiment_sha256='changed'), 'experiment'),
    (lambda r:r.update(status='failed'), 'failed'),
    (lambda r:r['optimizer_calls'].update(detail=4001), 'accounting'),
    (lambda r:r['optimizer_calls'].update(detail=4003), 'optimizer calls'),
    (lambda r:r['unique_new_updates_per_arm'].update(detail=4002), 'Unique'),
    (lambda r:r['measurement']['detail'].update(maximum_state_difference=1e-9), 'exact-resume'),
    (lambda r:r['measurement']['detail'].update(loss=1.1), 'Resume loss'),
    (lambda r:r['measurement']['detail'].update(seconds=float('nan')), 'duration'),
    (lambda r:r['frozen_modules']['baseline'].update(tensors=97), 'Frozen'),
    (lambda r:r['migrations']['baseline'].update(optimizer_exact=False), 'Migration'),
    (lambda r:r.update(spent_seconds=10801), 'spent_seconds'),
    (lambda r:r.update(estimated_units=4.01), 'estimated_units'),
    (lambda r:r.update(saved_common_steps=SAVE_STEPS[:-1]), 'Saved'),
])
def test_bad_report_evidence_cannot_validate(mutate, match):
    plan = plan_template(); report, updates = report_template(plan)
    mutate(report)
    with pytest.raises(ValueError, match=match):
        audit.validate_report(report, plan, 'experiment', updates)


def test_half_pair_is_counted_but_never_promoted():
    plan = plan_template(); report, updates = report_template(plan, 1002)
    partial = dict(arm='baseline', step=1003, comparable=False, path='baseline/partial-step-1003.pt')
    report['partial_checkpoints'] = [partial]; report['optimizer_calls']['baseline'] += 1
    audit.validate_report(report, plan, 'experiment', updates)
    partial['comparable'] = True
    with pytest.raises(ValueError, match='promoted'):
        audit.validate_report(report, plan, 'experiment', updates)


def test_budget_may_end_between_resume_measurements_without_claiming_both_passed():
    plan = plan_template(); report, updates = report_template(plan, 1001)
    del report['measurement']['detail']; del report['measurement']['both_arms_passed']
    report['optimizer_calls']['detail'] -= 1; report['replay_optimizer_calls']['detail'] = 0
    report['saved_common_steps'] = []
    audit.validate_report(report, plan, 'experiment', updates)
    report['measurement']['both_arms_passed'] = True
    with pytest.raises(ValueError, match='Both-arm'):
        audit.validate_report(report, plan, 'experiment', updates)


def test_full_assessments_require_train84_val64_and_source_consistency():
    plan = plan_template(); report, _ = report_template(plan); add_assessments(report)
    audit.validate_evaluations(report, plan, selection())
    report['evaluations'][-1]['report']['by_source']['val-15']['samples'] = 3
    with pytest.raises(ValueError, match='sample count'):
        audit.validate_evaluations(report, plan, selection())


def test_partial_assessment_is_only_allowed_last_at_budget_boundary():
    plan = plan_template(); report, _ = report_template(plan, 2000); add_assessments(report)
    report['evaluations'] = report['evaluations'][:-1]
    report['evaluations'][-1]['report'] = assessment('train', 2000, 17)
    audit.validate_evaluations(report, plan, selection())
    report['evaluations'].append(dict(arm='detail', common_step=2000, report=assessment('val',2000)))
    with pytest.raises(ValueError, match='Incomplete'):
        audit.validate_evaluations(report, plan, selection())


def test_assessment_prefix_cannot_substitute_another_source_or_report_bad_pooled_metric():
    plan = plan_template(); report, _ = report_template(plan, 999)
    report['evaluations'] = [dict(arm='baseline',common_step=999,report=assessment('train',999,1))]
    metric = report['evaluations'][0]['report']
    metric['by_source']['train-83'] = metric['by_source'].pop('train-0')
    with pytest.raises(ValueError, match='prefix'):
        audit.validate_evaluations(report, plan, selection())
    metric['by_source']['train-0'] = metric['by_source'].pop('train-83')
    metric['mse'] = .2
    with pytest.raises(ValueError, match='Pooled'):
        audit.validate_evaluations(report, plan, selection())


class MetadataCohort:
    def __init__(self):
        self.split='train'; self.length=16; self.binding_sha256='binding'
        self.rows={}; self.sequences=[]; self.by_source=defaultdict(list)
        self.cumulative=[]; self.source_cumulative={}; total=0
        counts=[3753]*83+[315320-3753*83]
        for i,count in enumerate(counts):
            source=f'train-{i}'; key=(source,'block'); windows=count-15
            self.rows[key]=dict(split='train',frames=count)
            self.sequences.append(dict(source_id=source,keys=[key],windows=windows))
            self.by_source[source].append(i); total+=windows; self.cumulative.append(total)
            self.source_cumulative[source]=[windows]
        self.sources=sorted(self.by_source)

    def window(self, *args):
        raise AssertionError('Artifact verification must never decode images')


def parent_state(contract, sampler):
    lr=.0003*(.1+.9*(1+math.cos(math.pi*(999-500)/(20000-500)))/2)
    model={f'lam.frozen-{i}':torch.tensor([float(i)]) for i in range(98)}
    model['tokenizer.weight']=torch.tensor([.5])
    rng=rng_state(); rng['cuda']=[torch.zeros(8,dtype=torch.uint8)]
    return dict(step=999,config={'size':128},contract=contract,model=model,best=float('inf'),
        optimizer={'state':{0:dict(step=torch.tensor(999.),exp_avg=torch.tensor([.1]),exp_avg_sq=torch.tensor([.2]))},
                   'param_groups':[dict(params=[0],lr=lr,initial_lr=.0003,weight_decay=.01)]},
        scheduler=dict(base_lrs=[.0003],last_epoch=999,_step_count=1000,_last_lr=[lr],lr_lambdas=[None]),
        scaler=dict(scale=1024.,growth_factor=2.,backoff_factor=.5,growth_interval=2000,_growth_tracker=999),
        sampler=sampler,rng=rng,spent_seconds=2000.,spent_units=.75,last_failure=None)


def checkpoint(parent, contract, sampler, step):
    state=copy.deepcopy(parent); state.update(step=step,contract=contract,sampler=sampler,spent_seconds=50.,spent_units=.05)
    progress=min(1,(step-500)/max(1,20000-500))
    factor=.1+.9*(1+math.cos(math.pi*progress))/2; lr=.0003*factor
    state['scheduler'].update(last_epoch=step,_step_count=step+1,_last_lr=[lr])
    state['optimizer']['param_groups'][0]['lr']=lr
    state['optimizer']['state'][0]['step'].fill_(step)
    state['scaler']['_growth_tracker']=step%2000
    state['model']['tokenizer.weight']+=(step-999)*.01
    return state


def test_arbitrary_budget_stop_preserves_original_scheduler_arithmetic():
    parent=parent_state({},{}); step=1887
    state=checkpoint(parent,{},parent['sampler'],step)
    rounded_differently=.0003*(.1+.9*(1+math.cos(math.pi*(step-500)/(20000-500)))/2)
    assert state['scheduler']['_last_lr'][0] != rounded_differently
    frozen={name:value for name,value in parent['model'].items() if name.startswith('lam.')}
    audit.validate_state(state,parent,{},parent['sampler'],step,dict(spent_seconds=100.,estimated_units=.1),frozen)


@pytest.fixture
def tiny_download(tmp_path,monkeypatch):
    root=tmp_path; run=root/'download'; run.mkdir(); (root/'continuation-inputs').mkdir()
    plan=plan_template(); cohort=MetadataCohort()
    sampler=MixedSampler(cohort).state_dict(); sampler['draws']=15985
    spec=dict(model={'size':128,'context':16},data_binding='binding.json',evaluation_selection='evaluation.json')
    # Model architecture is intentionally tiny; these tests validate evidence, never model quality.
    (root/'binding.json').write_text('{}'); (root/'evaluation.json').write_text(json.dumps(selection()))
    (root/'spec.json').write_text(json.dumps(spec)); cohort.binding_sha256=file_hash(root/'binding.json')
    prior=dict(arms={a:dict(spec='spec.json',spec_sha256=file_hash(root/'spec.json'),objective={'name':a}) for a in audit.ARMS},
               source_hashes={'implementation_sha256':'implementation'})
    (root/'prior.json').write_text(json.dumps(prior)); plan['parent_experiment']=dict(path='prior.json',sha256=file_hash(root/'prior.json'))
    parents={}
    for arm in audit.ARMS:
        contract=dict(stage='tokenizer',preflight=False,microbatch=4,accumulation=4,precision='fp16',device_type='cuda',
            torch_version='fixture-cuda',spec_sha256=file_hash(root/'spec.json'),binding_sha256=file_hash(root/'binding.json'),
            evaluation_sha256=file_hash(root/'evaluation.json'),implementation_sha256='implementation',
            comparison=dict(experiment_sha256=plan['parent_experiment']['sha256'],arm=arm,
                            objective=prior['arms'][arm]['objective'],source_hashes=prior['source_hashes']))
        state=parent_state(contract,sampler); state['config']=spec['model']; parents[arm]=state
        path=root/plan['parents'][arm]['path']; torch.save(state,path); plan['parents'][arm]['sha256']=file_hash(path)
        old=root/audit.ORIGINAL_RUN/arm/'last-common.pt'; old.parent.mkdir(parents=True); old.write_bytes(path.read_bytes())
    (root/'plan.json').write_text(json.dumps(plan)); experiment_sha=file_hash(root/'plan.json')
    report,updates=report_template(plan,1002); report['experiment_sha256']=experiment_sha; add_assessments(report)
    states,stats=audit.replay_sampling(cohort,sampler,plan['coverage_seed'],{999,1000,1001,1002})
    for arm in audit.ARMS:
        folder=run/arm; folder.mkdir(); report['coverage'][arm]=stats[1002]
        contract=audit.expected_contract(parents[arm],plan,experiment_sha,arm)
        for name,step in {'branch-start.pt':999,'measurement-boundary.pt':1000,'step-001001.pt':1001,
                          'last.pt':1002,'last-common.pt':1002}.items():
            torch.save(checkpoint(parents[arm],contract,states[step],step),folder/name)
    (run/'continuation-report.json').write_text(json.dumps(report))
    (run/'updates.jsonl').write_text('\n'.join(json.dumps(row) for row in updates)+'\n')
    monkeypatch.setattr(audit,'validate_experiment',lambda root,path:plan)
    monkeypatch.setattr(audit,'Cohort',lambda *args:cohort)
    return root,run,plan


def test_complete_artifact_audit_replays_metadata_and_preserves_all_bytes(tiny_download):
    root,run,plan=tiny_download
    before={p:file_hash(p) for p in root.rglob('*') if p.is_file()}
    result=audit.validate(root,run,'plan.json')
    assert result['status']=='validated' and result['run_status']=='budget_exhausted'
    assert result['branch_state_exact_to_parents'] and result['frozen_tensors']==98
    assert len(result['checkpoints'])==10 and result['coverage']['post_parent_draws']==48
    assert result['coverage']['fallback_draws']==0 and result['promotion_approved'] is False
    assert {p:file_hash(p) for p in before}==before


@pytest.mark.parametrize('fault', ['optimizer_nan','frozen','branch_weights','sampler','scheduler','scaler','rng','unknown_file','original'])
def test_corrupted_download_is_rejected(tiny_download,fault):
    root,run,plan=tiny_download
    path=run/'baseline'/('branch-start.pt' if fault=='branch_weights' else 'last.pt')
    state=audit.load(path)
    if fault=='optimizer_nan':state['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))
    if fault=='frozen':state['model']['lam.frozen-0'].add_(1)
    if fault=='branch_weights':state['model']['tokenizer.weight'].add_(1)
    if fault=='sampler':state['sampler']['coverage_index']+=1
    if fault=='scheduler':state['scheduler']['last_epoch']-=1
    if fault=='scaler':state['scaler']['scale']=65536.
    if fault=='rng':state['rng']['cuda'][0][0]=1
    if fault=='unknown_file':torch.save(state,run/'unreported.pt')
    elif fault=='original':(root/audit.ORIGINAL_RUN/'baseline/last-common.pt').write_bytes(b'changed')
    else:torch.save(state,path)
    with pytest.raises((ValueError,AssertionError)):
        audit.validate(root,run,'plan.json')


def test_replay_reaches_full_coverage_then_resumes_parent_mixed_state_without_images():
    cohort=MetadataCohort(); parent=MixedSampler(cohort).state_dict(); parent['draws']=15985
    states,stats=audit.replay_sampling(cohort,parent,20261110,{999,5000})
    assert stats[5000]['coverage_unique_frames']==315320 and stats[5000]['full_coverage']
    assert stats[5000]['post_parent_draws']==4001*16
    mixed=MixedSampler(cohort); mixed.load_state_dict(parent)
    for _ in range(stats[5000]['fallback_draws']):mixed.draw()
    audit.compare_exact(states[5000]['mixed'],mixed.state_dict())
