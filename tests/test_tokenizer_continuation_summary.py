import copy
import json

import pytest

from scripts import summarize_tokenizer_continuation as summary


def metric(first=.1, second=.3, complete=True):
    return dict(complete=complete, samples=3, sources=2,
                **{name:(first*2+second)/3 for name in summary.METRICS},
                by_source={'a':dict(samples=2, **{name:first for name in summary.METRICS}),
                           'b':dict(samples=1, **{name:second for name in summary.METRICS})})


def test_paired_source_signs_and_equal_source_do_not_reweight_by_window_count():
    result = summary.compare_metrics(metric(.1,.3), metric(.05,.4))['mse']
    assert result['pooled']['decrease'] == pytest.approx(0)
    assert result['equal_source']['reference'] == pytest.approx(.2)
    assert result['equal_source']['candidate'] == pytest.approx(.225)
    assert result['source_signs'] == dict(detail_lower=1, detail_higher=1, tied=0, unavailable=0,
                                         assessable=2,total_sources=2,detail_lower_fraction=.5)


def test_source_ties_missing_foreground_and_zero_reference_are_explicit():
    a,b=metric(0,.3),metric(0,.3+1e-13)
    a['foreground_mse']=b['foreground_mse']=None
    a['by_source']['b']['foreground_mse']=None
    b['by_source']['b']['foreground_mse']=None
    result=summary.compare_metrics(a,b)
    assert result['mse']['source_signs']['tied']==2
    assert result['foreground_mse']['source_signs']['unavailable']==1
    assert result['foreground_mse']['equal_source']['reference']==0
    assert result['foreground_mse']['equal_source']['percent_lower'] is None
    assert summary.change(0,.1)['percent_lower'] is None
    assert summary.change(.1,.05)['percent_lower']==pytest.approx(50)


@pytest.mark.parametrize('problem', ['partial','sources','source_samples','nan'])
def test_invalid_paired_populations_or_metrics_rejected(problem):
    a,b=metric(),metric()
    if problem=='partial':b['complete']=False
    if problem=='sources':b['by_source']['c']=b['by_source'].pop('b')
    if problem=='source_samples':b['by_source']['a']['samples']=1
    if problem=='nan':b['mse']=float('nan')
    with pytest.raises(ValueError):summary.compare_metrics(a,b)


@pytest.fixture
def inputs(tmp_path):
    run=tmp_path/'run';run.mkdir()
    report=dict(experiment_sha256='experiment',latest_common_step=5000,status='completed',
                coverage={arm:dict(full_coverage=True,coverage_unique_frames=315320) for arm in summary.ARMS},
                evaluations=[])
    for step in [999,5000]:
        for arm in summary.ARMS:
            for split in ['train','val']:
                factor=(1 if step==999 else .5)*(1 if arm=='baseline' else .8)*(1 if split=='train' else 2)
                report['evaluations'].append(dict(common_step=step,arm=arm,
                                                   report=dict(split=split,**metric(.1*factor,.3*factor))))
    audit=dict(status='validated',promotion_approved=False,quality_gates='unmeasured',
               experiment_sha256='experiment',latest_common_step=5000,run_status='completed',
               sampling_independently_replayed=True,parent_checkpoints_unchanged=True,
               frozen_lam_dynamics_exact=True,coverage=report['coverage']['baseline'])
    report_path=run/'continuation-report.json';audit_path=tmp_path/'audit.json'
    def save():
        report_path.write_text(json.dumps(report),encoding='utf-8')
        audit['files']=[dict(path=report_path.name,sha256=summary.sha(report_path))]
        audit_path.write_text(json.dumps(audit),encoding='utf-8')
    save()
    return report,audit,report_path,audit_path,save


def test_bound_summary_preserves_inputs_and_keeps_probe_gaps_descriptive(inputs):
    report,audit,report_path,audit_path,_=inputs
    hashes=(summary.sha(report_path),summary.sha(audit_path))
    result=summary.summarize(report_path,audit_path)
    assert len(result['paired_assessments'])==4 and result['excluded_assessments']==[]
    assert len(result['arm_changes_from_999'])==4
    assert all(row['metrics']['mse']['percent_lower']==pytest.approx(50) for row in result['arm_changes_from_999'])
    assert all(row['metrics']['mse']['validation_over_train']==pytest.approx(2) for row in result['train_validation_probe_differences'])
    assert result['quality_gates']=='unmeasured' and result['promotion_approved'] is False
    assert 'do not isolate' in result['policy']['train_validation']
    assert hashes==(summary.sha(report_path),summary.sha(audit_path))


def test_partial_assessment_is_excluded_instead_of_compared(inputs):
    report,audit,report_path,audit_path,save=inputs
    report['status']=audit['run_status']='budget_exhausted'
    report['evaluations'][-1]['report']['complete']=False
    save()
    result=summary.summarize(report_path,audit_path)
    assert len(result['paired_assessments'])==3
    assert result['excluded_assessments']==[dict(step=5000,split='val',reason='Missing or incomplete paired assessment')]
    assert len(result['train_validation_probe_differences'])==3


@pytest.mark.parametrize('problem',['bytes','experiment','coverage','promoted','not_replayed','duplicate'])
def test_stale_or_inconsistent_evidence_is_rejected(inputs,problem):
    report,audit,report_path,audit_path,save=inputs
    if problem=='bytes':
        report_path.write_text(report_path.read_text()+'\n')
    else:
        if problem=='experiment':audit['experiment_sha256']='different'
        if problem=='coverage':report['coverage']['detail']['full_coverage']=False
        if problem=='promoted':audit['promotion_approved']=True
        if problem=='not_replayed':audit['sampling_independently_replayed']=False
        if problem=='duplicate':report['evaluations'].append(copy.deepcopy(report['evaluations'][0]))
        save()
    with pytest.raises(ValueError):summary.summarize(report_path,audit_path)
