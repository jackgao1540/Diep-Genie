"""Extend an immutable, verified cohort with explicitly reviewed low-motion blocks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import copy
import numpy as np
from genie_diep.data import file_hash
from scripts.report_sequence_lengths import sequence_runs


def read(path,sha=None):
    if sha and file_hash(path)!=sha:raise ValueError('Evidence changed: '+str(path))
    return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]
def key(row):return row['source_id'],row['block_id']
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def resolve_native_names(screen, decision):
    result=copy.deepcopy(screen)
    overrides=decision.get('native_name_overrides',[])
    if not overrides:return result
    unresolved={i for i,d in enumerate(screen['detections']) if not d.get('native_name_allowed')}
    if {d['detection_index'] for d in overrides}!=unresolved or len(overrides)!=len(unresolved):
        raise ValueError('Incomplete native-name review')
    if set(screen['reasons'])!={'low_motion_review','text'}:raise ValueError('Unexpected screening override')
    result['original_screening_reasons']=list(screen['reasons'])
    result['native_name_overrides']=overrides
    for d in overrides:
        detection=result['detections'][d['detection_index']]
        if d['text']!=detection['text'] or d['frame']!=detection['frame'] or not d['reason']:
            raise ValueError('Native-name evidence mismatch')
        detection['original_native_name_allowed']=False
        detection['native_name_allowed']=True
    result['reasons']=['low_motion_review']
    return result


def validate_boundary(boundary, endpoints):
    lookup={key(r):r for r in endpoints}
    before=lookup.get((boundary['source_id'],boundary['before']))
    after=lookup.get((boundary['source_id'],boundary['after']))
    if before is None or after is None:raise ValueError('Boundary endpoint missing')
    if before['proxy_sha256']!=boundary['before_sha256'] or after['proxy_sha256']!=boundary['after_sha256']:raise ValueError('Boundary endpoint changed')
    if any(before[k]!=after[k] for k in ['video_sha256','crop','split','fps','source_fps']):raise ValueError('Boundary source mismatch')
    if before['start']>=after['start'] or abs(before['end']-after['start'])>=.6/before['source_fps']:raise ValueError('Boundary is not adjacent')
    return boundary['source_id'],boundary['before'],boundary['after']


def validate_additions(base,added,exclusions):
    identities={key(r) for r in base};by_source={}
    for row in base:by_source.setdefault(row['source_id'],[]).append(row)
    for r in added:
        if key(r) in identities:raise ValueError('Duplicate recovered identity')
        identities.add(key(r))
        if r['split'] not in {'train','val'}:raise ValueError('Reserved test forbidden')
        prior=by_source.get(r['source_id'])
        if not prior or any(r[k]!=prior[0][k] for k in ['video_sha256','split','crop','fps','source_fps']):raise ValueError('Source binding mismatch')
        if r['reasons']!=['low_motion_review'] or r['ocr_frames']!=r['frames']:raise ValueError('Unresolved screening')
        if any(not d.get('native_name_allowed') for d in r['detections']):raise ValueError('Unresolved text')
        if any(r['source_id']==e['source_id'] and r['start']<e['end']-1e-6 and r['end']>e['start']+1e-6 for e in exclusions):raise ValueError('Existing exclusion/hold')
        for old in prior:
            if set(r['source_frame_indices'])&set(old['source_frame_indices']):raise ValueError('Overlapping frames')
            if min(r['end'],old['end'])-max(r['start'],old['start'])>.6/r['source_fps']:raise ValueError('Overlapping time')
        by_source[r['source_id']].append(r)


def run(recipe_path,destination):
    recipe=read(recipe_path);out=Path(destination)
    if out.exists():raise FileExistsError('Choose a new immutable cohort')
    if recipe.get('review'):read(recipe['review'],recipe['review_sha256'])
    parent=Path(recipe['parent_cohort']);checkpoint=read(recipe['parent_checkpoint'])
    for name,sha in checkpoint['hashes'].items():
        if file_hash(parent/name)!=sha:raise ValueError('Parent cohort changed')
    base=rows(parent/'manifest.jsonl');prior_sequences=read(parent/'sequences.json')
    index=read(recipe['native_index'],recipe['native_index_sha256'])
    report=read(recipe['duplicate_report'],recipe['duplicate_report_sha256'])
    if report['base_manifest_sha256']!=checkpoint['hashes']['manifest.jsonl'] or index['base_manifest_sha256']!=report['base_manifest_sha256']:raise ValueError('Audit cohort mismatch')
    if report['exact_pairs']:raise ValueError('Unresolved exact duplicate')
    decisions=recipe['duplicate_decisions']
    if {d['index'] for d in decisions}!=set(range(len(report['near_runs']))) or len(decisions)!=len(report['near_runs']):raise ValueError('Incomplete duplicate review')
    if any(d['decision']!='distinct_gameplay' for d in decisions):raise ValueError('Unresolved near duplicate')
    for artifact in recipe['duplicate_review_evidence']:
        if file_hash(artifact['path'])!=artifact['sha256']:raise ValueError('Duplicate visual evidence changed')
    accepted_keys={(d['source_id'],d['block_id']) for d in recipe['accepted']}
    endpoints=base+[e['row'] for e in index['rows'] if key(e['row']) in accepted_keys]
    added=[];new_joins=set()
    for decision in recipe['accepted']:
        e=next(e for e in index['rows'] if key(e['row'])==(decision['source_id'],decision['block_id']))
        old=e['row'];screen_path=Path(old['recovery_screen']);screen=rows(screen_path)[0]
        if file_hash(screen_path)!=decision['rescreen_sha256']:raise ValueError('Rescreen changed')
        for field in ['video_sha256','crop','split','fps','source_fps','frames','source_frame_indices']:
            if old[field]!=screen[field]:raise ValueError('Recovery provenance mismatch')
        if not decision['all_native_frames_reviewed'] or not decision['ordinary_shooter'] or not decision['clean_gameplay']:raise ValueError('Incomplete visual review')
        for ev in [e['native_review']]+[b['evidence'] for b in e['boundaries']]:
            for artifact in ev['sheets']+[ev['context']]:
                if file_hash(artifact['path'])!=artifact['sha256']:raise ValueError('Visual evidence changed')
        if file_hash(old['media_path'])!=old['video_sha256'] or file_hash(old['proxy_path'])!=old['proxy_sha256']:raise ValueError('Recovery pixels changed')
        r=dict(resolve_native_names(screen,decision),block_id=old['block_id'],start=old['start'],end=old['end'],proxy_path=old['proxy_path'])
        added.append(r)
        expected={(b['source_id'],b['before'],b['after']) for b in e['boundaries']}
        supplied={(b['source_id'],b['before'],b['after']) for b in decision['approved_boundaries']}
        if expected!=supplied:raise ValueError('Incomplete boundary review')
        for b in e['boundaries']:
            new_joins.add(validate_boundary(b,endpoints))
    if len(added)!=report['additions'] or len({key(r) for r in added})!=len(added):raise ValueError('Addition audit mismatch')
    exclusions=read('configs/longform-cohort-v6.json')['exclude_intervals']
    validate_additions(base,added,exclusions)
    joins={(s['source_id'],a,b) for s in prior_sequences for a,b in zip(s['blocks'],s['blocks'][1:])}|set(new_joins)
    out.mkdir(parents=True);all_rows=list(base);summary=read(parent/'summary.json')
    coverage=summary['coverage_color_proxies'];recipe_sha=file_hash(recipe_path)
    for r in added:
        relative=Path('clips')/r['source_id']/(r['block_id']+'.npz')
        row=dict(r,original_proxy_path=r['proxy_path'],proxy_path=relative.as_posix(),proxy=relative.name,
                 reasons=[],screening_reasons=r['reasons'],historical_screening_reasons=['freeze'],
                 disposition='curation_accepted',quality_approved=True,class_review_required=False,
                 curation_accepted=True,training_approved=False,recovery_recipe=recipe_path,recovery_recipe_sha256=recipe_sha)
        all_rows.append(row)
        with np.load(r['proxy_path'],allow_pickle=False) as data:frames=data['frames'].astype(np.int16)
        red,green,blue=frames[...,0],frames[...,1],frames[...,2];c=coverage[r['source_id']]
        for name,mask in [('yellow',(red>150)&(green>130)&(blue<140)),('blue',(blue>120)&(blue-red>25)&(blue-green>15)),('red',(red>140)&(red-green>35)&(red-blue>20))]:
            c[name+'_visible_frames']+=int((mask.sum(axis=(1,2))>=4).sum())
        c['frames']+=len(frames)
    lookup={key(r):r for r in added}
    for r in all_rows:
        target=out/r['proxy_path'];target.parent.mkdir(parents=True,exist_ok=True)
        source=Path(lookup[key(r)]['proxy_path']) if key(r) in lookup else parent/r['proxy_path']
        if file_hash(source)!=r['proxy_sha256']:raise ValueError('Source proxy changed')
        shutil.copy2(source,target)
        if file_hash(target)!=r['proxy_sha256']:raise ValueError('Copied proxy changed')
    runs=sequence_runs(all_rows,lambda a,b:(a['source_id'],a['block_id'],b['block_id']) in joins)
    seconds=Counter()
    for r in all_rows:seconds[r['split']]+=r['frames']/r['fps']
    final_by_key={key(r):r for r in all_rows}
    assert all(final_by_key[key(r)]==r for r in base)
    summary.update(clips=len(all_rows),source_count=len({r['source_id'] for r in all_rows}),seconds_by_split=dict(seconds),
                   curation_accepted_hours=sum(seconds.values())/3600,longest_sequence_seconds=max(s['seconds'] for s in runs),
                   seconds_in_runs_at_least={str(n):sum(s['seconds'] for s in runs if s['seconds']>=n) for n in [10,20,30,60]},
                   recipe=recipe_path,recipe_sha256=recipe_sha,parent_cohort=str(parent),
                   recovery=dict(added_blocks=len(added),added_seconds=sum(r['frames']/r['fps'] for r in added),new_joins=len(new_joins),
                                 inherited_parent_evidence=True),training_approved=False,corpus_complete=False)
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_rows),encoding='utf-8',newline='\n')
    write(out/'sequences.json',runs);write(out/'summary.json',summary)
    result=dict(cohort=out.as_posix(),parent_cohort=parent.as_posix(),hashes={n:file_hash(out/n) for n in ['summary.json','manifest.jsonl','sequences.json']},
                accepted_blocks=len(all_rows),accepted_hours=summary['curation_accepted_hours'],source_count=summary['source_count'],
                seconds_by_split=dict(seconds),added_blocks=len(added),added_seconds=sum(r['frames']/r['fps'] for r in added),parent_rows_unchanged=True,
                reserved_test_unchanged=True,new_joins=len(new_joins),training_approved=False)
    write(recipe.get('output_checkpoint','configs/longform-seventh-checkpoint-v1.json'),result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('recipe');p.add_argument('out');a=p.parse_args();run(a.recipe,a.out)
