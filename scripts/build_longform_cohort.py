"""Freeze a reviewed curation cohort; final corpus/training readiness is separate."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import shutil
import numpy as np
from genie_diep.data import file_hash
from genie_diep.curation_reviews import load_class_review
from scripts.report_sequence_lengths import sequence_runs


def read(path,sha=None):
    if sha and file_hash(path)!=sha:raise ValueError('Evidence changed: '+str(path))
    return json.loads(Path(path).read_text(encoding='utf-8'))


def indices(ranges,total):
    result=set()
    for start,end in ranges:
        if not 0<=start<=end<total:raise ValueError('Review range outside index')
        for i in range(start,end+1):
            if i in result:raise ValueError('Overlapping review ranges')
            result.add(i)
    return result


def key(row):return row['source_id'],row['block_id']


def validate_duplicate_exclusions(rows,report,confirmed,excluded):
    """A confirmed near-run is resolved only when one whole candidate side is withheld."""
    if report['exact_pairs']:raise ValueError('Exact duplicate resolution required')
    flags=report['near_runs']
    if len(confirmed)!=len(set(confirmed)):raise ValueError('Repeated confirmed duplicate')
    for index in confirmed:
        if not isinstance(index,int) or not 0<=index<len(flags):raise ValueError('Invalid confirmed duplicate index')
        flag=flags[index];resolved=False
        for side in ('a','b'):
            overlapping={key(r) for r in rows if r['source_id']==flag['source_'+side]
                and r['start']<=flag['end_'+side] and r['end']>flag['start_'+side]}
            if overlapping and overlapping<=excluded:resolved=True
        if not resolved:raise ValueError('Confirmed duplicate still present on both sides')


def validate(recipe):
    rows=read(recipe['candidate_index'],recipe['candidate_index_sha256'])['rows']
    by_key={key(r):r for r in rows}
    if len(rows)!=len(by_key):raise ValueError('Repeated source block')
    decisions={}
    for p in recipe['class_reviews']:
        for r in load_class_review(p):
            if key(r) in decisions:raise ValueError('Repeated class decision')
            decisions[key(r)]=r
    for r in rows:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',r['source_id']) or not re.fullmatch(r'\d+_\d+',r['block_id']):raise ValueError('Unsafe artifact identifier')
        d=decisions.get(key(r))
        if not d or not d['class_candidate'] or d['proxy_sha256']!=r['proxy_sha256']:raise ValueError('Missing matching class review')
        if r['reasons'] or r['ocr_frames']!=r['frames']:raise ValueError('Incomplete clean screening')
        if any(not d.get('native_name_allowed',False) for d in r['detections']):raise ValueError('Unresolved text detection')
        if file_hash(r['proxy_path'])!=r['proxy_sha256']:raise ValueError('Candidate changed')
    sources={s['id']:s for p in recipe['source_recipes'] for s in read(p)['sources']}
    source_splits={};hash_splits={};last_end={}
    for r in sorted(rows,key=lambda r:(r['source_id'],r['start'])):
        s=sources[r['source_id']]
        if s['video_sha256']!=r['video_sha256'] or s['crop']!=r['crop'] or s['split']!=r['split']:raise ValueError('Source provenance mismatch')
        for mapping,identity in [(source_splits,r['source_id']),(hash_splits,r['video_sha256'])]:
            if identity in mapping and mapping[identity]!=r['split']:raise ValueError('Source split leakage')
            mapping[identity]=r['split']
        if r['start']<last_end.get(r['source_id'],-1)-.6/r['source_fps']:raise ValueError('Overlapping source time')
        last_end[r['source_id']]=r['end']
    for source in source_splits:
        s=sources[source]
        if file_hash(s['media_path'])!=s['video_sha256']:raise ValueError('Original media changed')
    audit=read(recipe['visual_audit']);sample=read(Path(audit['review'])/'index.json',audit['index_sha256'])['rows']
    reviewed=indices(audit['reviewed_ranges'],len(sample));rejected={int(i) for i in audit['rejected_indices']}
    if reviewed!=set(range(len(sample))) or len(reviewed)<300 or not rejected<=reviewed:raise ValueError('Incomplete sample audit')
    if len(rejected)/len(reviewed)>.01:raise ValueError('Sample quality threshold failed')
    excluded=set()
    for i,r in enumerate(sample):
        if key(r) not in by_key or by_key[key(r)]['proxy_sha256']!=r['proxy_sha256']:raise ValueError('Sample not bound to cohort')
        if i in rejected:excluded.add(key(r))
    for r in rows:
        for e in recipe['exclude_intervals']:
            if r['source_id']==e['source_id'] and r['start']<e['end']-1e-6 and r['end']>e['start']+1e-6:excluded.add(key(r))
    dup=read(recipe['duplicate_review']);report=read(dup['report'],dup['report_sha256'])
    if dup['index_sha256']!=recipe['candidate_index_sha256']:raise ValueError('Duplicate audit cohort mismatch')
    validate_duplicate_exclusions(rows,report,dup['confirmed_duplicate_indices'],excluded)
    if indices(dup['reviewed_near_run_ranges'],len(report['near_runs']))!=set(range(len(report['near_runs']))):raise ValueError('Unreviewed duplicate flags')
    if report['saturated_hash_buckets']:raise ValueError('Duplicate lookup saturated')
    bound=read(recipe['boundary_review']);bi=read(bound['index'],bound['index_sha256'])
    if bi['source_index_sha256']!=recipe['candidate_index_sha256']:raise ValueError('Boundary cohort mismatch')
    inspected=indices(bound['reviewed_ranges'],len(bi['pairs']));bad=set(bound['rejected_indices']);joins=set()
    if inspected!=set(range(len(bi['pairs']))) or not bad<=inspected:raise ValueError('Incomplete boundary review')
    for i,p in enumerate(bi['pairs']):
        for side,block in [('before',p['before']),('after',p['after'])]:
            r=by_key.get((p['source_id'],block))
            if not r or r['proxy_sha256']!=p[side+'_sha256']:raise ValueError('Boundary hash mismatch')
        if i not in bad:joins.add((p['source_id'],p['before'],p['after']))
    accepted=[r for r in rows if key(r) not in excluded]
    runs=sequence_runs(accepted,lambda a,b:(a['source_id'],a['block_id'],b['block_id']) in joins)
    return accepted,runs,dict(sample_clips=len(sample),sample_rejected=len(rejected),excluded_blocks=len(excluded),
        original_candidates=len(rows),sources=[sources[s] for s in sorted(source_splits)])


def run(args):
    out=Path(args.out)
    if out.exists():raise FileExistsError('Choose a new immutable cohort directory')
    recipe=read(args.recipe);rows,runs,evidence=validate(recipe)
    out.mkdir(parents=True);counts=Counter();coverage=defaultdict(Counter)
    for r in rows:
        relative=Path('clips')/r['source_id']/(r['block_id']+'.npz');target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(r['proxy_path'],target)
        if file_hash(target)!=r['proxy_sha256']:raise ValueError('Copied proxy mismatch')
        r=dict(r,original_proxy_path=r['proxy_path'],proxy_path=relative.as_posix(),curation_accepted=True,training_approved=False)
        counts[r['split']]+=r['frames']/r['fps']
        with np.load(target,allow_pickle=False) as data:
            frames=data['frames'].astype(np.int16)
            # Color coverage is a proxy, not semantic object ground truth.
            red,green,blue=frames[...,0],frames[...,1],frames[...,2]
            for name,mask in [('yellow',(red>150)&(green>130)&(blue<140)),('blue',(blue>120)&(blue-red>25)&(blue-green>15)),('red',(red>140)&(red-green>35)&(red-blue>20))]:
                coverage[r['source_id']][name+'_visible_frames']+=int((mask.sum(axis=(1,2))>=4).sum())
            coverage[r['source_id']]['frames']+=len(frames)
        with (out/'manifest.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
    durations=[r['seconds'] for r in runs]
    summary=dict(curation_accepted_hours=sum(counts.values())/3600,clips=len(rows),source_count=len({r['source_id'] for r in rows}),
        seconds_by_split=counts,longest_sequence_seconds=max(durations,default=0),
        seconds_in_runs_at_least={str(n):sum(d for d in durations if d>=n) for n in (10,20,30,60)},
        coverage_color_proxies=coverage,evidence=evidence,recipe=args.recipe,recipe_sha256=file_hash(args.recipe),
        corpus_complete=False,training_approved=False,scope=recipe['scope'])
    (out/'sequences.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('evidence','coverage_color_proxies')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('recipe');p.add_argument('out');run(p.parse_args())
