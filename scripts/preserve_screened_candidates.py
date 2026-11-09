"""Inventory and verify every saved long-form screen; preserve low-motion previews.

Originals and existing proxies stay in place. Exact screening evidence is copied
into an immutable local registry. Nothing here changes cohort membership.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from genie_diep.data import file_hash


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def content_id(row):
    binding={k:row[k] for k in ('video_sha256','crop','source_frame_indices','fps')}
    return hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')


def snapshot_class_evidence(root,destination):
    """An additive, independently bound snapshot; do not rewrite completed banks."""
    root=Path(root).resolve();out=Path(destination)
    if out.exists():raise FileExistsError('Choose a new class evidence snapshot')
    out.mkdir(parents=True);evidence=[];seen=set()
    def copy(path):
        path=path.resolve()
        if not path.is_relative_to(root):raise ValueError('Evidence outside project')
        if path in seen:return
        seen.add(path);sha=file_hash(path);target=out/(sha+path.suffix)
        shutil.copyfile(path,target)
        if file_hash(target)!=sha:raise ValueError('Class snapshot changed')
        evidence.append(dict(original=path.relative_to(root).as_posix(),snapshot=target.name,sha256=sha))
    for path in sorted((root/'configs').glob('*class*.json')):
        copy(path);decision=read(path)
        if 'review' in decision:
            index=root/decision['review']/'index.json'
            if index.exists():
                if not decision.get('invalidated_reason') and decision.get('index_sha256')!=file_hash(index):
                    raise ValueError('Class decision index changed')
                copy(index)
            elif not decision.get('invalidated_reason'):raise FileNotFoundError(index)
    result=dict(training_approved=False,evidence=evidence)
    write(out/'index.json',result);return result


def decode(media,row):
    targets=row['source_frame_indices']
    if len(targets)!=row['frames'] or targets!=sorted(set(targets)):
        raise ValueError('Invalid retained timestamps')
    cap=cv2.VideoCapture(str(media));cap.set(cv2.CAP_PROP_POS_FRAMES,targets[0])
    cursor=int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)));crops=[];context=None
    try:
        for i,target in enumerate(targets):
            if cursor>target:raise ValueError('Decoder seek overshot target')
            while cursor<target:
                if not cap.grab():raise ValueError('Decode failed')
                cursor+=1
            ok,frame=cap.read();cursor+=1
            if not ok:raise ValueError('Decode failed')
            x,y,w,h=row['crop']
            if min(x,y)<0 or min(w,h)<=0 or x+w>frame.shape[1] or y+h>frame.shape[0]:raise ValueError('Invalid crop')
            crops.append(cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2RGB))
            if i==len(targets)//2:context=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    finally:cap.release()
    return np.stack(crops),context


def run(root,destination,render=True):
    root=Path(root).resolve();out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Choose a new preservation checkpoint')
    out.mkdir(parents=True);(out/'evidence').mkdir();(out/'low-motion').mkdir()
    entries={};artifacts={};sources={};snapshots=[];errors=[];screen_count=record_count=0
    def artifact(path,expected,kind):
        path=path.resolve();key=str(path)
        if not path.is_relative_to(root):raise ValueError('Artifact outside project')
        if key not in artifacts:
            actual=file_hash(path) if path.exists() else None
            artifacts[key]=dict(path=path.relative_to(root).as_posix(),sha256=actual,bytes=path.stat().st_size if actual else 0,kind=kind)
        if artifacts[key]['sha256']!=expected:
            errors.append(dict(path=artifacts[key]['path'],expected=expected,actual=artifacts[key]['sha256']))
        return artifacts[key]
    for progress in sorted((root/'runs').rglob('blocks.jsonl')):
        config_path=progress.with_name('configuration.json')
        if not config_path.exists():continue
        config=read(config_path);source=config['source'];screen_count+=1
        media=root/source['media_path'];source_artifact=artifact(media,config['video_sha256'],'original')
        sources[(source['id'],config['video_sha256'])]=source_artifact
        for path in (config_path,progress):
            sha=file_hash(path);target=out/'evidence'/(sha+path.suffix)
            if not target.exists():shutil.copyfile(path,target)
            if file_hash(target)!=sha:raise ValueError('Evidence snapshot mismatch')
            snapshots.append(dict(original=path.relative_to(root).as_posix(),snapshot=target.relative_to(out).as_posix(),sha256=sha))
        for line in progress.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);record_count+=1
            if row['video_sha256']!=config['video_sha256'] or row['source_id']!=source['id']:raise ValueError('Screen/source mismatch')
            cid=content_id(row)
            entry=entries.setdefault(cid,dict(id=cid,source_id=row['source_id'],video_sha256=row['video_sha256'],media_path=source_artifact['path'],observations=[]))
            observation=dict(screen=progress.parent.relative_to(root).as_posix(),screen_version=config.get('screen_version',1),record=row)
            entry['observations'].append(observation)
            if row.get('proxy'):
                artifact(progress.parent/row['proxy'],row['proxy_sha256'],'proxy')
        if screen_count%25==0:print(json.dumps(dict(screens=screen_count,records=record_count)),flush=True)
    active_path=root/'data/curated-longform-v6/manifest.jsonl'
    active_rows=[json.loads(s) for s in active_path.read_text().splitlines()] if active_path.exists() else []
    active={content_id(r) for r in active_rows};test_sources={r['source_id'] for r in active_rows if r['split']=='test'}
    low_motion=[];rejected_count=0
    for entry in entries.values():
        obs=entry['observations'];records=[o['record'] for o in obs]
        entry['historical_screen_rejection']=any(r['reasons'] for r in records)
        rejected_count+=entry['historical_screen_rejection']
        entry['in_current_cohort']=entry['id'] in active
        entry['preservation_required']=True;entry['training_approved']=False
        entry['quality_status']='unassessed_by_preservation_inventory'
        entry['model_eligibility']='not_assessed; retain separate class review decisions'
        entry['reserved_test']=entry['source_id'] in test_sources or any(r['split']=='test' for r in records)
        if any(set(r['reasons'])&{'freeze','low_motion_review'} for r in records):
            best=max(obs,key=lambda o:o['screen_version']);row=best['record']
            queue=dict(id=entry['id'],source_id=entry['source_id'],block_id=row['block_id'],split=row['split'],
                       start=row['start'],seconds=row['frames']/row['fps'],media_path=entry['media_path'],
                       video_sha256=row['video_sha256'],crop=row['crop'],source_frame_indices=row['source_frame_indices'],
                       original_reasons=row['reasons'],screen=best['screen'],
                       historically_motion_only=any(set(r['reasons'])<={'freeze','low_motion_review'} and r['reasons'] for r in records),
                       in_current_cohort=entry['in_current_cohort'],reserved_test=entry['reserved_test'],
                       status='preserved_pending_review',quality_approved=False,training_approved=False)
            if render and not entry['reserved_test'] and not errors:
                crops,context=decode(root/entry['media_path'],row)
                folder=out/'low-motion'/entry['id'];folder.mkdir()
                proxy=np.stack([cv2.resize(f,(128,128),interpolation=cv2.INTER_AREA) for f in crops])
                np.savez_compressed(folder/'frames.npz',frames=proxy)
                w,h=crops.shape[2],crops.shape[1]
                sheet=Image.new('RGB',(8*w,((len(crops)+7)//8)*(h+18)),(245,245,240));draw=ImageDraw.Draw(sheet)
                for i,f in enumerate(crops):
                    x=i%8*w;y=i//8*(h+18);sheet.paste(Image.fromarray(f),(x,y+18));draw.text((x+4,y+2),str(row['source_frame_indices'][i]),fill='black')
                sheet.save(folder/'retained-frames.png');Image.fromarray(context).save(folder/'context.png')
                queue['review_artifacts']={p.name:dict(path=p.relative_to(out).as_posix(),sha256=file_hash(p)) for p in folder.iterdir()}
            low_motion.append(queue)
    # Snapshot class/quality decisions too; their status is not replaced by this inventory.
    for path in sorted((root/'configs').glob('*.json')):
        if not any(word in path.name for word in ['review','audit','class']):continue
        sha=file_hash(path);target=out/'evidence'/(sha+'.json');shutil.copyfile(path,target)
        snapshots.append(dict(original=path.relative_to(root).as_posix(),snapshot=target.relative_to(out).as_posix(),sha256=sha))
    with (out/'registry.jsonl').open('w',encoding='utf-8',newline='\n') as handle:
        for entry in sorted(entries.values(),key=lambda e:e['id']):handle.write(json.dumps(entry)+'\n')
    write(out/'artifacts.json',list(artifacts.values()));write(out/'evidence.json',snapshots)
    write(out/'low-motion-review.json',dict(training_approved=False,rows=low_motion))
    snapshot_class_evidence(root,out/'class-evidence')
    summary=dict(schema_version=1,screen_directories=screen_count,screen_records=record_count,
                 unique_content_blocks=len(entries),historically_rejected_blocks=rejected_count,
                 originals_verified=sum(a['kind']=='original' and a['sha256'] is not None for a in artifacts.values()),
                 existing_proxies_verified=sum(a['kind']=='proxy' and a['sha256'] is not None for a in artifacts.values()),
                 low_motion_blocks=len(low_motion),low_motion_only_blocks=sum(q['historically_motion_only'] for q in low_motion),
                 low_motion_already_in_cohort=sum(q['in_current_cohort'] for q in low_motion),
                 low_motion_previews=sum('review_artifacts' in q for q in low_motion),
                 reserved_test_previews=0,integrity_errors=errors,training_approved=False,
                 preservation='Originals and existing proxies hash-checked in place; screening and review metadata snapshotted; no deletion, repair or cohort promotion.',
                 hashes={name:file_hash(out/name) for name in ['registry.jsonl','artifacts.json','evidence.json','low-motion-review.json']})
    write(out/'summary.json',summary);print(json.dumps(summary),flush=True)
    if errors:raise ValueError('Preserved inventory has missing/changed artifacts; inspect summary')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.');p.add_argument('--out',required=True)
    p.add_argument('--no-render',action='store_true');a=p.parse_args();run(a.root,a.out,not a.no_render)
