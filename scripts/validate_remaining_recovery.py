"""Validate the v8 recovery, preserved queue, split protection and review evidence."""
from collections import Counter
from pathlib import Path
from genie_diep.data import file_hash
from scripts.build_stationary_recovery import read, rows, key, write, validate_boundary
from scripts.serve_footage_review import Library


def validate():
    checkpoint=read('configs/longform-eighth-checkpoint-v1.json')
    parent=read('configs/longform-seventh-checkpoint-v1.json')
    recipe=read('configs/longform-cohort-v8.json')
    review=read(recipe['review'],recipe['review_sha256'])
    original=read(review['index'],review['index_sha256'])
    native=read(review['native_index'],review['native_index_sha256'])
    final=read(recipe['native_index'],recipe['native_index_sha256'])
    duplicate=read(recipe['duplicate_report'],recipe['duplicate_report_sha256'])
    for binding in [parent,checkpoint]:
        for name,sha in binding['hashes'].items():assert file_hash(Path(binding['cohort'])/name)==sha
    base={key(r):r for r in rows(Path(parent['cohort'])/'manifest.jsonl')}
    records=rows(Path(checkpoint['cohort'])/'manifest.jsonl');current={key(r):r for r in records}
    assert len(current)==len(records)==10414
    assert all(current[k]==r for k,r in base.items())
    added=set(current)-set(base)
    assert added=={key(r) for r in recipe['accepted']} and len(added)==28
    assert all(current[k]['split']=='train' for k in added)
    assert {r['source_id'] for r in records}=={r['source_id'] for r in base.values()}
    assert 'eGJEVXrV8rs' not in {r['source_id'] for r in records}
    seconds=Counter()
    for r in records:
        assert file_hash(Path(checkpoint['cohort'])/r['proxy_path'])==r['proxy_sha256']
        seconds[r['split']]+=r['frames']/r['fps']
    assert dict(seconds)==checkpoint['seconds_by_split']=={'train':31512,'val':3768,'test':6376}
    assert {key(r):r for r in records if r['split']=='test'}=={k:r for k,r in base.items() if r['split']=='test'}
    assert len(review['decisions'])==len(original['remaining'])==84
    assert {d['queue_index'] for d in review['decisions']}=={r['queue_index'] for r in original['remaining']}
    decisions={d['queue_index']:d for d in review['decisions']}
    assert Counter(d['decision'] for d in decisions.values())==review['decision_counts']
    assert {key(d) for d in decisions.values() if d['decision']=='accepted_v8'}==added
    bank=Path('data/preserved-candidates-v1')
    assert file_hash(bank/'low-motion-review.json')==original['queue_sha256']
    sources=set()
    for r in original['remaining']:
        d=decisions[r['queue_index']]
        assert d['preservation_id']==r['id'] and key(d)==key(r)
        if r['reserved_test']:assert d['decision']=='reserved_test'
        if r['media_path'] not in sources:
            assert file_hash(r['media_path'])==r['video_sha256'];sources.add(r['media_path'])
        for a in r.get('review_artifacts',{}).values():
            assert file_hash(bank/a['path'])==a['sha256']
    for artifact in original['sheets']+recipe['duplicate_review_evidence']:
        assert file_hash(artifact['path'])==artifact['sha256']
    assert len(native['rows'])==43
    for e in native['rows']:
        assert e['row']['split']!='test'
        assert file_hash(e['row']['recovery_screen'])==e['rescreen_sha256']
        for a in e['native_review']['sheets']+[e['native_review']['context']]:assert file_hash(a['path'])==a['sha256']
    assert len(duplicate['near_runs'])==792 and not duplicate['exact_pairs']
    assert {d['index'] for d in recipe['duplicate_decisions']}==set(range(792))
    expected=set()
    for e in final['rows']:
        for b in e['boundaries']:
            expected.add(validate_boundary(b,records))
            for a in b['evidence']['sheets']+[b['evidence']['context']]:assert file_hash(a['path'])==a['sha256']
    assert len(expected)==33
    sequences=read(Path(checkpoint['cohort'])/'sequences.json')
    members=[(s['source_id'],b) for s in sequences for b in s['blocks']]
    assert len(members)==len(set(members))==len(current) and set(members)==set(current)
    def joins(items):return {(s['source_id'],a,b) for s in items for a,b in zip(s['blocks'],s['blocks'][1:])}
    assert joins(sequences)==joins(read(Path(parent['cohort'])/'sequences.json'))|expected
    summary=read(Path(checkpoint['cohort'])/'summary.json')
    assert summary['recipe_sha256']==file_hash('configs/longform-cohort-v8.json')
    assert not summary['training_approved'] and not summary['corpus_complete']
    binding=read('configs/baseline-data-v8.json')
    assert binding['cohort']==checkpoint['cohort'] and binding['hashes']==checkpoint['hashes']
    library=Library(Path.cwd());old=Library(Path.cwd(),cohort=parent['cohort'])
    assert Counter(r['status'] for r in library.public.values())=={'accepted':8820,'held':160,'rejected':1}
    assert all(r['split']!='test' for r in library.public.values())
    assert all(library.feedback.get(k)==v for k,v in old.feedback.items())
    result=dict(validation='passed',blocks=len(current),added_seconds=112,retained_seconds=sum(seconds.values()),
        all_proxy_hashes_verified=True,parent_rows_unchanged=True,reserved_test_unchanged=True,
        candidate_decisions=84,native_reviewed_candidates=43,accepted_native_frames=sum(e['native_review']['native_frames'] for e in final['rows']),
        new_joins=len(expected),preserved_feedback=len(old.feedback),training_approved=False)
    write('runs/remaining-low-motion-v1/checkpoint-validation.json',result)
    print(result)


if __name__=='__main__':validate()
