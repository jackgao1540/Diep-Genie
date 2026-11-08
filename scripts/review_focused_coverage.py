"""Local targeted coverage triage and dense review; never grants training approval.

Scores rank candidates only. Optical flow, color contours and barrel estimates
are fallible visual proxies, not action labels or semantic corpus statistics.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from scripts.report_longform_coverage import read, digest, key
from genie_diep.visual_probes import background_shift, barrel_probe


def candidates(frame):
    rgb = frame.astype(np.int16)
    blue = ((rgb[:,:,2]-rgb[:,:,0]>35)&(rgb[:,:,2]-rgb[:,:,1]>35)).astype(np.uint8)
    pentagons = []
    for contour in cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(contour)
        if area < 8:
            continue
        approx = cv2.approxPolyDP(contour, .045*cv2.arcLength(contour, True), True)
        _,_,w,h = cv2.boundingRect(contour)
        if 4 <= len(approx) <= 6 and min(w,h)/max(w,h) > .55:
            pentagons.append(area)
    gray = frame.mean(-1)
    neutral = (frame.max(-1).astype(int)-frame.min(-1)<12)&(gray>100)&(gray<195)
    wall_area = 0
    for contour in cv2.findContours(neutral.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
        _,_,w,h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if max(w,h)>50 and min(w,h)>12 and area/(w*h)>.7:
            wall_area = max(wall_area,area)
    return max(pentagons,default=0), wall_area


def scan(args):
    cv2.setNumThreads(1)
    root, out = Path(args.cohort), Path(args.out)
    if out.exists():raise FileExistsError(out)
    rows = [json.loads(l) for l in (root/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    results = []
    for r in rows:
        if r['split']=='test':continue
        path = root/r['proxy_path']
        if digest(path)!=r['proxy_sha256']:raise ValueError('Changed proxy')
        with np.load(path,allow_pickle=False) as data:frames=data['frames']
        if frames.shape!=(40,128,128,3):raise ValueError('Unexpected proxy shape')
        areas = [candidates(frames[i]) for i in [0,10,20,30,39]]
        record = dict(source_id=r['source_id'],block_id=r['block_id'],split=r['split'],start=r['start'],
            proxy_sha256=r['proxy_sha256'],mean_delta=r['metrics']['mean_delta'],
            pentagon_area=max(a[0] for a in areas),wall_area=max(a[1] for a in areas),
            stable_background_candidate=False,aim_sweep=None)
        if r['metrics']['mean_delta']<.025:
            offsets=[0,10,20,30,39]
            flows=[background_shift(frames[a],frames[b]) for a,b in zip(offsets,offsets[1:])]
            valid=[f for f in flows if f['dx'] is not None]
            if len(valid)>=3:
                record['max_background_step']=float(max(np.hypot(f['dx'],f['dy']) for f in valid))
                record['stable_background_candidate']=record['max_background_step']<1.
                if record['stable_background_candidate']:
                    poses=[barrel_probe(frames[i]) for i in offsets]
                    angles=[p['angle'] for p in poses if p['angle'] is not None and p['confidence']>=.5]
                    if len(angles)>=3:
                        record['aim_sweep']=float(np.ptp(np.unwrap(angles)))
        results.append(record)
        if len(results)%500==0:print(json.dumps(dict(scanned=len(results))),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(manifest_sha256=digest(root/'manifest.jsonl'),rows=results,
        warning=__doc__,test_pixels_reviewed=False,training_approved=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(scanned=len(results),stable_candidates=sum(r['stable_background_candidate'] for r in results))))


def render(args):
    cv2.setNumThreads(1)
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    request=read(args.request)
    root=Path(args.cohort)
    if request['manifest_sha256']!=digest(root/'manifest.jsonl'):raise ValueError('Cohort changed')
    recipe=read('configs/longform-cohort-v5.json')
    sources={s['id']:s for p in recipe['source_recipes'] for s in read(p)['sources']}
    manifest={key(r):r for r in [json.loads(l) for l in (root/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]}
    entries=[];verified=set();seen=set()
    out.mkdir(parents=True)
    for i,request_row in enumerate(request['rows']):
        identity=(request_row['source_id'],request_row['block_id'])
        if identity in seen:raise ValueError('Duplicate review request')
        seen.add(identity)
        accepted=identity in manifest
        if accepted:r=manifest[identity]
        else:
            matches=[json.loads(l) for l in Path(request_row['screen'],'blocks.jsonl').read_text(encoding='utf-8').splitlines()]
            matches=[r for r in matches if key(r)==identity]
            if len(matches)!=1:raise ValueError('Ambiguous rejected record')
            r=matches[0]
        if r['split'] not in {'train','val'}:raise ValueError('Reserved test forbidden')
        source=sources[r['source_id']]
        if r['video_sha256']!=source['video_sha256'] or r['crop']!=source['crop']:raise ValueError('Provenance mismatch')
        if r['source_id'] not in verified:
            if digest(source['media_path'])!=r['video_sha256']:raise ValueError('Original changed')
            verified.add(r['source_id'])
        if accepted and digest(root/r['proxy_path'])!=r['proxy_sha256']:raise ValueError('Proxy changed')
        cap=cv2.VideoCapture(source['media_path']);x,y,w,h=r['crop']
        sheet=Image.new('RGB',(1600,1060),'white');draw=ImageDraw.Draw(sheet)
        draw.text((4,4),f"#{i} {r['source_id']} {r['start']:.3f}s {r['split']} accepted={accepted} | all 40 retained timestamps, row-major at 10 FPS",fill='black')
        draw.text((4,20),'Targets: '+', '.join(request_row['targets'])+' | source crop at 200px; controls unknown',fill='black')
        for j,fi in enumerate(r['source_frame_indices']):
            cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,frame=cap.read()
            if not ok:raise RuntimeError('Decode failed')
            crop=cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2RGB)
            sheet.paste(Image.fromarray(crop).resize((200,200)),(j%8*200,60+j//8*200))
        # Full original context is especially useful for a near-static rejection.
        cap.set(cv2.CAP_PROP_POS_FRAMES,r['source_frame_indices'][20]);ok,frame=cap.read();cap.release()
        if not ok:raise RuntimeError('Context decode failed')
        context=out/f'context-{i:03}.png';cv2.imwrite(str(context),frame)
        path=out/f'dense-{i:03}.png';sheet.save(path)
        entries.append(dict(review_index=i,source_id=r['source_id'],block_id=r['block_id'],split=r['split'],
            start=r['start'],seconds=r['frames']/r['fps'],accepted=accepted,targets=request_row['targets'],
            reasons=r['reasons'],video_sha256=r['video_sha256'],proxy_sha256=r.get('proxy_sha256'),
            source_frame_indices=r['source_frame_indices'],crop=r['crop'],sheet=path.name,sheet_sha256=digest(path),
            context=context.name,context_sha256=digest(context)))
        print(f'Rendered {i+1}/{len(request["rows"])}',flush=True)
    (out/'index.json').write_text(json.dumps(dict(request_sha256=digest(args.request),
        manifest_sha256=request['manifest_sha256'],rows=entries,test_pixels_reviewed=False,
        training_approved=False,warning=__doc__),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['scan','render']);p.add_argument('--cohort',default='data/curated-longform-v5')
    p.add_argument('--out',required=True);p.add_argument('--request')
    args=p.parse_args()
    scan(args) if args.mode=='scan' else render(args)
