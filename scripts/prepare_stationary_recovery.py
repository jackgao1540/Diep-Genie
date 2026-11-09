"""Prepare native-rate evidence and incremental duplicate checks for reviewed recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from genie_diep.data import file_hash
from scripts.audit_longform_duplicates import fingerprint,near_frames,matching_runs


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def candidates():
    checkpoint=read('configs/preservation-checkpoint-v1.json')
    if file_hash(checkpoint['rescreen_material'])!=checkpoint['rescreen_material_sha256']:raise ValueError('Rescreen evidence changed')
    material=read(checkpoint['rescreen_material'])['rows'];result=[]
    for d in checkpoint['priority_reviews']:
        if d['recovery_status']!='visual_candidate':continue
        r=next(r for r in material if r['source_id']==d['source_id'])
        if file_hash(r['rescreen'])!=r['rescreen_sha256']:raise ValueError('Rescreen changed')
        screened=rows(r['rescreen'])[0]
        if screened['reasons']!=['low_motion_review'] or screened['ocr_frames']!=40:raise ValueError('Unresolved screening flag')
        old=next(x for x in rows(Path(r['original_screen'])/'blocks.jsonl') if x['block_id']==d['block_id'])
        config=read(Path(r['original_screen'])/'configuration.json')
        proxy=(Path(r['rescreen']).parent/screened['proxy']).as_posix()
        if file_hash(proxy)!=r['proxy_sha256'] or file_hash(config['source']['media_path'])!=r['video_sha256']:raise ValueError('Pixels changed')
        result.append(dict(old,proxy_path=proxy,proxy_sha256=r['proxy_sha256'],recovery_screen=r['rescreen'],
                           media_path=config['source']['media_path'],recovery_decision=d))
    return result


def native_sheet(media,crop,first,last,out,prefix):
    cap=cv2.VideoCapture(media);cap.set(cv2.CAP_PROP_POS_FRAMES,first)
    actual=int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    if actual!=first:raise ValueError('Native seek mismatch')
    frames=[];full=[];x,y,w,h=crop
    for fi in range(first,last):
        ok,frame=cap.read()
        if not ok:raise ValueError('Native decode failed')
        frames.append(cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2RGB))
        if fi in {first,(first+last)//2,last-1}:full.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    cap.release();paths=[]
    for start in range(0,len(frames),40):
        subset=frames[start:start+40];sheet=Image.new('RGB',(8*w,5*(h+16)),(245,245,240));draw=ImageDraw.Draw(sheet)
        for j,f in enumerate(subset):
            px=j%8*w;py=j//8*(h+16);sheet.paste(Image.fromarray(f),(px,py+16));draw.text((px+3,py),str(first+start+j),fill='black')
        path=out/f'{prefix}-{start//40:02}.png';sheet.save(path);paths.append(dict(path=path.as_posix(),sha256=file_hash(path)))
    context=Image.new('RGB',(full[0].shape[1]*len(full),full[0].shape[0]))
    for i,f in enumerate(full):context.paste(Image.fromarray(f),(i*f.shape[1],0))
    path=out/f'{prefix}-context.png';context.save(path)
    delta=np.abs(np.diff(np.stack(frames).astype(np.int16),axis=0)).mean((1,2,3))
    return dict(first_frame=first,last_frame_exclusive=last,native_frames=len(frames),exact_repeated_transitions=int((delta==0).sum()),
                max_mean_absolute_change=float(delta.max()),sheets=paths,context=dict(path=path.as_posix(),sha256=file_hash(path)))


def prepare(out):
    out=Path(out)
    if out.exists():raise FileExistsError('Choose a new recovery evidence directory')
    out.mkdir(parents=True);added=candidates();base=rows('data/curated-longform-v6/manifest.jsonl')
    for r in base:r['proxy_path']=(Path('data/curated-longform-v6')/r['proxy_path']).as_posix()
    evidence=[]
    for i,r in enumerate(added):
        first=r['source_frame_indices'][0];last=round(r['end']*r['source_fps'])
        review=native_sheet(r['media_path'],r['crop'],first,last,out,f'clip-{i}')
        boundaries=[]
        for other in base:
            if other['source_id']!=r['source_id']:continue
            if abs(other['end']-r['start'])<.6/r['source_fps'] or abs(r['end']-other['start'])<.6/r['source_fps']:
                before,after=(other,r) if other['start']<r['start'] else (r,other)
                join=round(before['end']*r['source_fps'])
                boundary=native_sheet(r['media_path'],r['crop'],join-6,join+6,out,f'boundary-{i}-{len(boundaries)}')
                boundaries.append(dict(before=before['block_id'],after=after['block_id'],source_id=r['source_id'],
                                       before_sha256=before['proxy_sha256'],after_sha256=after['proxy_sha256'],evidence=boundary))
        evidence.append(dict(row=r,native_review=review,boundaries=boundaries))
    write(out/'native-index.json',dict(base_manifest_sha256=file_hash('data/curated-longform-v6/manifest.jsonl'),rows=evidence))
    targets=[];exact={};matches=[];pairs=[]
    for i,r in enumerate(added):
        with np.load(r['proxy_path'],allow_pickle=False) as d:video=d['frames']
        exact[hashlib.sha256(str(video.shape).encode()+video.tobytes()).hexdigest()]=i
        for f in range(0,40,5):
            code,small=fingerprint(video[f]);targets.append((i,code,small,r['source_frame_indices'][f]/r['source_fps']))
    for bi,b in enumerate(base+added):
        with np.load(b['proxy_path'],allow_pickle=False) as d:video=d['frames']
        if file_hash(b['proxy_path'])!=b['proxy_sha256']:raise ValueError('Comparison proxy changed')
        digest=hashlib.sha256(str(video.shape).encode()+video.tobytes()).hexdigest()
        if digest in exact and bi!=len(base)+exact[digest]:pairs.append(dict(base_index=bi,addition_index=exact[digest]))
        for f in range(0,40,5):
            code,small=fingerprint(video[f]);time=b['source_frame_indices'][f]/b['source_fps']
            for ai,target,pixels,t in targets:
                a=added[ai]
                if bi==len(base)+ai:continue
                if b['source_id']==a['source_id'] and abs(time-t)<8:continue
                if (code^target).bit_count()>6 or not near_frames(small,pixels):continue
                matches.append(dict(source_a=b['source_id'],source_b=a['source_id'],time_a=time,time_b=t,split_a=b['split'],split_b=a['split']))
        if (bi+1)%1000==0:print(json.dumps(dict(duplicate_comparisons=bi+1)),flush=True)
    report=dict(base_manifest_sha256=file_hash('data/curated-longform-v6/manifest.jsonl'),base_clips=len(base),additions=len(added),
                exact_pairs=pairs,near_runs=matching_runs(matches),raw_matches=len(matches),sample_fps=2,
                method='Every addition compared against every v6 clip and other additions; 2 FPS foreground-aware near matching and whole-clip exact hashes. No hash bucket truncation. Same-source offsets under 8s skipped as adjacent gameplay.',
                limitations='Can miss transformed reuse; near matches are triage, not proof of reused recording. No new reserved-test visual review.')
    write(out/'duplicates.json',report);write(out/'duplicate-index.json',dict(rows=base+added))
    print(json.dumps(dict(native_clips=len(evidence),boundary_pairs=sum(len(e['boundaries']) for e in evidence),exact_pairs=len(pairs),near_runs=len(report['near_runs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();prepare(a.out)
