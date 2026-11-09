"""Resumable local, every-retained-frame screening of long gameplay intervals.

Outputs are CANDIDATES, never automatically training-approved. Original media,
timestamps and crop provenance remain available; proxies do not fix model size.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from genie_diep.data import file_hash
from genie_diep.screen_captions import intersects_crop, text_candidate, score_caption_candidate
from genie_diep.native_text import native_name


def player_present(bgr):
    """Palette-independent centered colored body triage, NOT a tank-class label."""
    h,w=bgr.shape[:2]
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    mask=((hsv[:,:,1]>100)&(hsv[:,:,2]>100)).astype(np.uint8)
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if not (h*w*.0008<=area<=h*w*.06):continue
        if np.linalg.norm(centers[i]-[w/2,h/2])>min(h,w)*.16:continue
        if min(bw,bh)/max(bw,bh)<.55:continue
        if area/(bw*bh)<.48:continue
        # Yellow food cannot stand in for the player. Custom palettes need review.
        hue=float(np.median(hsv[:,:,0][labels==i]))
        if 17<hue<40:continue
        return True
    return False


def cheap_reasons(frames, motion_policy='review'):
    if motion_policy not in ('review','legacy'):raise ValueError('Unknown motion policy')
    reasons=[]
    if not all(player_present(f) for f in frames):reasons.append('no_center_body')
    small=np.stack([cv2.resize(f,(64,64),interpolation=cv2.INTER_AREA) for f in frames])
    differences=np.abs(np.diff(small.astype(np.float32),axis=0)).mean((1,2,3))/255
    if differences.max()>.12:reasons.append('cut_or_flash')
    if differences.mean()<.0004:reasons.append('freeze' if motion_policy=='legacy' else 'low_motion_review')
    repeated=np.all(np.diff(np.stack(frames).astype(np.int16),axis=0)==0,axis=(1,2,3))
    streak=longest=0
    for same in repeated:
        streak=streak+1 if same else 0;longest=max(longest,streak)
    hsv=np.stack([cv2.cvtColor(f,cv2.COLOR_BGR2HSV) for f in small])
    neutral=((hsv[:,:,:,1]<45)&(hsv[:,:,:,2]>115)&(hsv[:,:,:,2]<250)).mean((1,2))
    if neutral.min()<.30:reasons.append('nonstandard_or_obscured_background')
    sharpness=[cv2.Laplacian(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var() for f in frames]
    if min(sharpness)<8:reasons.append('blur_or_fade')
    return reasons,dict(mean_delta=float(differences.mean()),max_delta=float(differences.max()),
                        min_neutral=float(neutral.min()),min_sharpness=float(min(sharpness)),
                        exact_repeat_fraction=float(repeated.mean()),longest_exact_repeat_frames=longest+1)


def crop_frame(frame,crop,padding):
    x,y,w,h=crop;fh,fw=frame.shape[:2]
    if min(x,y)<0 or min(w,h)<=0 or x+w>fw or y+h>fh:raise ValueError('Crop outside source')
    left=max(0,x-padding);top=max(0,y-padding)
    return frame[y:y+h,x:x+w],frame[top:min(fh,y+h+padding),left:min(fw,x+w+padding)],np.array([x-left,y-top])


def screen_source(source,args,engine,recipe_sha):
    motion_policy=getattr(args,'motion_policy','review')
    if motion_policy not in ('review','legacy'):raise ValueError('Unknown motion policy')
    caption_policy=getattr(args,'caption_policy','legacy')
    if caption_policy not in ('legacy','score-captions'):raise ValueError('Unknown caption policy')
    if caption_policy!='legacy' and getattr(args,'previous_screen',[]):
        raise ValueError('Stronger caption policy cannot reuse earlier negative screens')
    if motion_policy=='review' and getattr(args,'previous_screen',[]):
        raise ValueError('Low-motion review and rejected-proxy preservation require a fresh screen')
    candidate_fn=score_caption_candidate if caption_policy=='score-captions' else text_candidate
    out=Path(args.out)/source['id'];out.mkdir(parents=True,exist_ok=True)
    media=Path(source['media_path'])
    source_sha=file_hash(media)
    if source_sha!=source['video_sha256']:raise ValueError('Source changed after visual review')
    policy=getattr(args,'text_policy','strict')
    configuration=dict(recipe_sha256=recipe_sha,source=source,video_sha256=source_sha,
                       fps=args.fps,block_seconds=args.block_seconds,padding=args.padding,
                       proxy_size=args.proxy_size,screen_version=1)
    if policy!='strict':configuration.update(screen_version=2,text_policy=policy)
    if caption_policy!='legacy':configuration.update(screen_version=3,caption_policy=caption_policy,text_policy=policy)
    if motion_policy=='review':configuration.update(screen_version=4,motion_policy=motion_policy,preserve_rejected_proxies=True)
    config=out/'configuration.json'
    if config.exists() and json.loads(config.read_text())!=configuration:
        raise ValueError('Screen configuration changed: choose a new output directory')
    config.write_text(json.dumps(configuration,indent=2),encoding='utf-8')
    progress=out/'blocks.jsonl';done={}
    if progress.exists():
        for line in progress.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);done[r['block_id']]=r
            if (not r['reasons'] or motion_policy=='review') and (not r.get('proxy') or not (out/r['proxy']).exists() or file_hash(out/r['proxy'])!=r['proxy_sha256']):
                raise ValueError('Completed candidate proxy missing or changed')
    prior={}
    for previous_root in getattr(args,'previous_screen',[]):
        previous=Path(previous_root)/source['id'];previous_config=previous/'configuration.json'
        if not previous_config.exists():continue
        pc=json.loads(previous_config.read_text(encoding='utf-8'))
        if pc.get('screen_version')!=1:raise ValueError('Only unchanged strict-v1 screening can be reused')
        for key in ('source','video_sha256','fps','block_seconds','padding','proxy_size'):
            if pc[key]!=configuration[key]:raise ValueError('Prior screening geometry/source mismatch')
        lines=(previous/'blocks.jsonl').read_text(encoding='utf-8').splitlines()
        for line in lines:
            r=json.loads(line)
            if r['block_id'] in prior:raise ValueError('Duplicate previous block')
            prior[r['block_id']]=(previous,r)
    cap=cv2.VideoCapture(str(media));source_fps=cap.get(cv2.CAP_PROP_FPS)
    if source_fps<args.fps:raise ValueError('Source FPS below retained FPS')
    frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));duration=frame_count/source_fps
    block_frames=round(args.fps*args.block_seconds)
    if block_frames<2 or abs(block_frames/args.fps-args.block_seconds)>1e-6:raise ValueError('Nonintegral block')
    started=time.monotonic();counts=Counter();crop=source['crop'];previous_end=0
    for interval_index,(start,end) in enumerate(source['intervals']):
        if not (0<=start<end<=duration+.1) or start<previous_end:raise ValueError('Invalid/overlapping intervals')
        previous_end=end
        # Sequential decode within each interval; no seek for each OCR frame.
        for block_index in range(math.floor((end-start)/args.block_seconds)):
            block_id=f'{interval_index:03}_{block_index:06}'
            if block_id in done:continue
            if block_id in prior:
                previous,old=prior[block_id]
                # Reuse strict negatives and unchanged cheap rejections. A text
                # rejection stopped OCR early, so it MUST be screened in full.
                if 'text' not in old['reasons']:
                    record=dict(old,reused_strict_screen=str(previous))
                    if not old['reasons']:
                        if old['ocr_frames']!=block_frames:raise ValueError('Incomplete prior OCR')
                        path=previous/old['proxy']
                        if file_hash(path)!=old['proxy_sha256']:raise ValueError('Changed prior proxy')
                        shutil.copy2(path,out/old['proxy'])
                    with progress.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n');f.flush()
                    done[block_id]=record;continue
            block_start=start+block_index*args.block_seconds
            first_frame=round(block_start*source_fps)
            if abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-first_frame)>1:
                cap.set(cv2.CAP_PROP_POS_FRAMES,first_frame)
            frames=[];padded=[];full_frames=[];timestamps=[];origin=None
            targets=[round((block_start+j/args.fps)*source_fps) for j in range(block_frames)]
            current=int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            for target in targets:
                while current<target:
                    if not cap.grab():raise RuntimeError('Decode failed while advancing')
                    current+=1
                ok,frame=cap.read();current+=1
                if not ok:raise RuntimeError(f'Decode failed: {source["id"]} {target}')
                a,b,origin=crop_frame(frame,crop,args.padding)
                frames.append(a.copy());padded.append(b.copy());full_frames.append(frame);timestamps.append(target/source_fps)
            reasons,metrics=cheap_reasons(frames,motion_policy);detections=[];ocr_frames=0
            low_motion_only=reasons==['low_motion_review']
            if not reasons or low_motion_only:
                for i,pad in enumerate(padded):
                    result=engine(pad);ocr_frames+=1
                    if result.boxes is None:continue
                    for box,txt,score in zip(result.boxes,result.txts,result.scores):
                        box=box-origin
                        if intersects_crop(box,crop[2],crop[3]) and candidate_fn(txt,score,box):
                            allowed=policy=='native-names' and native_name(full_frames[i],box+np.array(crop[:2]))
                            detections.append(dict(frame=i,text=txt,score=float(score),box=box.tolist(),native_name_allowed=bool(allowed)))
                    if any(not d['native_name_allowed'] for d in detections):
                        if 'text' not in reasons:reasons.append('text')
                        if not low_motion_only:break
            record=dict(block_id=block_id,source_id=source['id'],split=source['split'],
                        start=timestamps[0],end=block_start+args.block_seconds,
                        source_frame_indices=targets,source_fps=source_fps,fps=args.fps,
                        crop=crop,video_sha256=source_sha,frames=block_frames,metrics=metrics,
                        reasons=reasons,detections=detections,ocr_frames=ocr_frames,
                        text_policy=policy,
                        training_approved=False,class_review_required=True)
            if motion_policy=='review':
                record.update(disposition='manual_review' if reasons else 'screen_candidate',
                              quality_approved=False,preservation_required=True,motion_policy=motion_policy)
            if not reasons or motion_policy=='review':
                proxy=np.stack([cv2.cvtColor(cv2.resize(f,(args.proxy_size,args.proxy_size),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB) for f in frames])
                path=out/f'{block_id}.npz';np.savez_compressed(path,frames=proxy)
                record.update(proxy=path.name,proxy_sha256=file_hash(path))
            with progress.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n');f.flush()
            done[block_id]=record;counts.update(reasons or ['candidate'])
            if len(done)%25==0:
                print(json.dumps(dict(source=source['id'],blocks=len(done),new_counts=dict(counts),seconds=round(time.monotonic()-started,1))),flush=True)
    cap.release()
    accepted=[r for r in done.values() if not r['reasons']]
    summary=dict(source_id=source['id'],blocks=len(done),candidate_blocks=len(accepted),
                 candidate_seconds=sum(r['frames']/r['fps'] for r in accepted),
                 reasons=dict(Counter(reason for r in done.values() for reason in r['reasons'])),
                 training_approved=False,warning='Automatic triage only; class, overlays, edits, duplicates and sample audit remain required.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


def run(args):
    from rapidocr import RapidOCR
    engine=RapidOCR(params={'Global.log_level':'error','EngineConfig.onnxruntime.intra_op_num_threads':args.threads,
                           'EngineConfig.onnxruntime.inter_op_num_threads':1,'Det.limit_side_len':384})
    recipe=json.loads(Path(args.recipe).read_text(encoding='utf-8'))
    if args.shards<1 or not 0<=args.shard<args.shards:raise ValueError('Invalid shard')
    ids=[s['id'] for s in recipe['sources']]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate source in recipe')
    for i,source in enumerate(recipe['sources']):
        if i%args.shards==args.shard:screen_source(source,args,engine,file_hash(args.recipe))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('recipe');p.add_argument('out')
    p.add_argument('--fps',type=float,default=10);p.add_argument('--block-seconds',type=float,default=4)
    p.add_argument('--padding',type=int,default=24);p.add_argument('--proxy-size',type=int,default=128)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--shards',type=int,default=1);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--text-policy',choices=['strict','native-names'],default='strict')
    p.add_argument('--caption-policy',choices=['legacy','score-captions'],default='legacy')
    p.add_argument('--previous-screen',nargs='*',default=[])
    p.add_argument('--motion-policy',choices=['review','legacy'],default='review',
                   help='Review preserves every proxy and treats low change as a review flag; legacy is only for historical reproduction.')
    run(p.parse_args())
