"""Verify the real incremental recovery against its unchanged parent cohort."""
from collections import Counter
from pathlib import Path
import json
from genie_diep.data import file_hash
from scripts.build_stationary_recovery import read, rows, key
from scripts.serve_footage_review import Library


def validate():
    checkpoint=read('configs/longform-seventh-checkpoint-v1.json')
    parent=read('configs/longform-sixth-checkpoint-v1.json')
    recipe=read('configs/longform-cohort-v7.json')
    for binding in [parent,checkpoint]:
        for name,sha in binding['hashes'].items():
            assert file_hash(Path(binding['cohort'])/name)==sha
    base={key(r):r for r in rows(Path(parent['cohort'])/'manifest.jsonl')}
    records=rows(Path(checkpoint['cohort'])/'manifest.jsonl')
    current={key(r):r for r in records}
    assert len(current)==len(records)==10386
    assert all(current[k]==r for k,r in base.items())
    added=set(current)-set(base)
    assert added=={key(r) for r in recipe['accepted']} and len(added)==3
    assert all(current[k]['split']=='train' for k in added)
    assert {r['source_id'] for r in records}=={r['source_id'] for r in base.values()}
    assert 'eGJEVXrV8rs' not in {r['source_id'] for r in records}
    seconds=Counter()
    for r in records:
        assert file_hash(Path(checkpoint['cohort'])/r['proxy_path'])==r['proxy_sha256']
        seconds[r['split']]+=r['frames']/r['fps']
    assert dict(seconds)==checkpoint['seconds_by_split']=={'train':31400,'val':3768,'test':6376}
    sequences=read(Path(checkpoint['cohort'])/'sequences.json')
    members=[(s['source_id'],b) for s in sequences for b in s['blocks']]
    assert len(members)==len(set(members))==len(current) and set(members)==set(current)
    def joins(items):return {(s['source_id'],a,b) for s in items for a,b in zip(s['blocks'],s['blocks'][1:])}
    expected={(b['source_id'],b['before'],b['after']) for d in recipe['accepted'] for b in d['approved_boundaries']}
    assert joins(sequences)==joins(read(Path(parent['cohort'])/'sequences.json'))|expected
    assert len(expected)==2
    summary=read(Path(checkpoint['cohort'])/'summary.json')
    assert summary['recipe_sha256']==file_hash('configs/longform-cohort-v7.json')
    assert summary['curation_accepted_hours']==11.54 and not summary['training_approved']
    binding=read('configs/baseline-data-v7.json')
    assert binding['cohort']==checkpoint['cohort'] and binding['hashes']==checkpoint['hashes']
    library=Library(Path.cwd(),cohort=checkpoint['cohort'])
    assert Counter(r['status'] for r in library.public.values())=={'accepted':8792,'held':160,'rejected':1}
    assert all(r['split']!='test' for r in library.public.values())
    old=Library(Path.cwd(),cohort=parent['cohort'])
    assert all(library.feedback.get(k)==v for k,v in old.feedback.items())
    print(json.dumps(dict(validation='passed',blocks=len(current),added_seconds=12,retained_seconds=sum(seconds.values()),
                          all_proxy_hashes_verified=True,parent_rows_unchanged=True,reserved_test_unchanged=True,
                          new_joins=len(expected),preserved_feedback=len(old.feedback))))


if __name__=='__main__':validate()
