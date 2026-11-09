"""Release local readiness for the bounded GPU pilot, never model quality approval."""
from collections import Counter
from pathlib import Path
from genie_diep.baseline import implementation_hash
from genie_diep.cohort import Cohort
from genie_diep.data import file_hash
from scripts.build_stationary_recovery import read,rows,key,write,validate_boundary


def run():
    c=read('configs/longform-ninth-checkpoint-v1.json');parent=read('configs/longform-eighth-checkpoint-v1.json')
    recipe=read('configs/longform-cohort-v9.json');review=read(recipe['review'],recipe['review_sha256'])
    native=read(recipe['native_index'],recipe['native_index_sha256']);duplicate=read(recipe['duplicate_report'],recipe['duplicate_report_sha256'])
    for bound in [c,parent]:
        for name,sha in bound['hashes'].items():assert file_hash(Path(bound['cohort'])/name)==sha
    base={key(r):r for r in rows(Path(parent['cohort'])/'manifest.jsonl')};current=rows(Path(c['cohort'])/'manifest.jsonl');lookup={key(r):r for r in current}
    assert len(lookup)==len(current)==10419 and all(lookup[k]==r for k,r in base.items())
    additions=set(lookup)-set(base);assert additions=={key(r) for r in recipe['accepted']} and len(additions)==5
    seconds=Counter()
    for r in current:
        assert file_hash(Path(c['cohort'])/r['proxy_path'])==r['proxy_sha256'];seconds[r['split']]+=r['frames']/r['fps']
    assert dict(seconds)=={'train':31532,'val':3768,'test':6376}
    assert all(lookup[k]['split']=='train' for k in additions)
    assert not duplicate['exact_pairs'] and len(duplicate['near_runs'])==324
    assert {d['index'] for d in recipe['duplicate_decisions']}==set(range(324))
    assert review['class_name']=='Sprayer' and review['ordinary_shooter']
    for a in recipe['duplicate_review_evidence']:assert file_hash(a['path'])==a['sha256']
    joins=set()
    for e in native['rows']:
        for b in e['boundaries']:
            joins.add(validate_boundary(b,current))
            for a in b['evidence']['sheets']+[b['evidence']['context']]:assert file_hash(a['path'])==a['sha256']
        for a in e['native_review']['sheets']+[e['native_review']['context']]:assert file_hash(a['path'])==a['sha256']
        assert file_hash(e['row']['recovery_screen'])==e['rescreen_sha256']
    assert len(joins)==5
    def pairs(sequences):return {(s['source_id'],a,b) for s in sequences for a,b in zip(s['blocks'],s['blocks'][1:])}
    assert pairs(read(Path(c['cohort'])/'sequences.json'))==pairs(read(Path(parent['cohort'])/'sequences.json'))|joins
    spec=read('configs/training-baseline-v1.json');execution=read('runs/training-readiness-v1/execution/report.json')
    assert execution['status']=='passed' and execution['implementation_sha256']==implementation_hash()
    assert execution['spec_sha256']==file_hash('configs/training-baseline-v1.json')
    assert execution['binding_sha256']==file_hash('configs/baseline-data-v9.json')
    assert execution['evaluation_sha256']==file_hash('configs/baseline-evaluation-v1.json')
    assert all(s['exact_resume'] and s['frozen_modules_unchanged'] and s['full_model_parameters']==2623654 for s in execution['stages'])
    fidelity=read('configs/baseline-fidelity-review-v1.json');index=read(fidelity['index'],fidelity['index_sha256'])
    assert fidelity['all_eight_sheets_reviewed'] and fidelity['source_resize_pixel_exact']
    for a in index['sheets']+fidelity['supplemental_context_evidence']:assert file_hash(a['path'])==a['sha256']
    evaluation=read('configs/baseline-evaluation-v1.json');val=Cohort('.','configs/baseline-data-v9.json','val',16);train=Cohort('.','configs/baseline-data-v9.json','train',16)
    assert len(evaluation['validation'])==64 and len({r['source_id'] for r in evaluation['validation']})==16
    for r in evaluation['validation']:val.locate(r)
    for r in evaluation['training_calibration']:train.locate(r)
    assert set(val.sources).isdisjoint(train.sources)
    result=dict(schema_version=1,ready_for_gpu_pilot=True,ready_for_unbounded_training=False,training_approved=False,
        first_stage='Bounded tokenizer GPU measurement: maximum 2 compute units / 1 allocated hour; preserve 50-unit reserve',
        spec_sha256=file_hash('configs/training-baseline-v1.json'),binding_sha256=file_hash('configs/baseline-data-v9.json'),
        implementation_sha256=implementation_hash(),execution_report='runs/training-readiness-v1/execution/report.json',execution_report_sha256=file_hash('runs/training-readiness-v1/execution/report.json'),
        fidelity_review='configs/baseline-fidelity-review-v1.json',fidelity_review_sha256=file_hash('configs/baseline-fidelity-review-v1.json'),
        retained_seconds=sum(seconds.values()),retained_blocks=len(current),reserved_test_unchanged=True,
        recovered_blocks_cumulative=36,validation_windows=64,validation_sources=16,train_calibration_sources=len(train.sources),
        checks=dict(dataset_hashes=True,class_hold_resolved=True,checked_sequence_loading=True,split_isolation=True,real_cpu_finite_execution=True,
            exact_cpu_resume=True,frozen_stages=True,all_action_codebooks_exercised=True,source_fidelity=True),
        next_gate='Authorize allocation and enter live rate/balance, then measure GPU FP16/memory/throughput and resume. Stop on failure; no automatic longer training.',
        quality_gates=dict(G1='unmeasured',G2='unmeasured',G3='unmeasured'),
        later_stages='Action/dynamics promotion still requires scored reconstruction, training-only semantic calibration and held-out interventions/rollouts.')
    write('configs/training-readiness-v1.json',result);print(result)


if __name__=='__main__':run()
