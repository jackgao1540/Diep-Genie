"""Local OCR triage on original-resolution crops, before tiny training resize.

Text detections include names/HUD and can miss edited captions. A negative
result never approves a clip; combine with class/cut and visual review.
"""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw


def text_candidate(text,score,box):
    # Food triangles/squares were misread as "44" and "70" in the pilot.
    # Numeric-only detections must not preferentially remove food-rich clips.
    box=np.asarray(box)
    return bool(sum(c.isalpha() for c in text)>=2 and np.ptp(box[:,0])>=16
                and np.ptp(box[:,1])>=8 and score>=.5)


def intersects_crop(box,width,height):
    box=np.asarray(box)
    return bool(box[:,0].max()>0 and box[:,0].min()<width and box[:,1].max()>0 and box[:,1].min()<height)


def run(args):
    from rapidocr import RapidOCR
    engine=RapidOCR(params={'Global.log_level':'error','EngineConfig.onnxruntime.intra_op_num_threads':2,
        'EngineConfig.onnxruntime.inter_op_num_threads':1,'Det.limit_side_len':384})
    root=Path(args.data);raw=Path(args.raw);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for p in sorted(root.glob('*.jsonl')) for line in p.read_text(encoding='utf-8').splitlines() if line]
    if args.previous_screen:
        previous=json.loads(Path(args.previous_screen).read_text(encoding='utf-8'))
        prior={r['path']:r for r in previous['rows']}
        if set(prior)!={r['path'] for r in rows}:raise ValueError('Previous screen must cover the entire candidate manifest')
        rows=[r for r in rows if not prior[r['path']]['suspicious']]
    if args.limit:rows=rows[:args.limit]
    results=[];cap=None;source=None;samples=[];started=time.monotonic()
    for i,row in enumerate(rows):
        if row['source_id']!=source:
            if cap is not None:cap.release()
            source=row['source_id'];files=[p for ext in ('mp4','webm','mkv') if (p:=raw/f'{source}.{ext}').exists()]
            if not files:raise FileNotFoundError(source)
            cap=cv2.VideoCapture(str(files[0]))
        indices=range(row['shape'][0]) if args.every_frame else sorted(set((0,row['shape'][0]//2,row['shape'][0]-1)))
        detections=[];sampled=0
        for index in indices:
            timestamp=row['start']+index/row['fps'];cap.set(cv2.CAP_PROP_POS_MSEC,timestamp*1000);ok,frame=cap.read()
            if not ok:raise RuntimeError(f'Cannot inspect {source} at {timestamp}')
            origin=np.zeros(2)
            if row.get('crop'):
                x,y,w,h=row['crop'];left=max(0,x-args.padding);top=max(0,y-args.padding)
                ocr_frame=frame[top:min(frame.shape[0],y+h+args.padding),left:min(frame.shape[1],x+w+args.padding)]
                origin=np.array([x-left,y-top]);frame=frame[y:y+h,x:x+w]
            else:ocr_frame=frame
            sampled+=1;result=engine(ocr_frame)
            if result.boxes is None:continue
            for box,txt,score in zip(result.boxes,result.txts,result.scores):
                box=box-origin
                if not intersects_crop(box,frame.shape[1],frame.shape[0]):continue
                suspicious=text_candidate(txt,score,box)
                detections.append(dict(frame=index,time=timestamp,text=txt,score=float(score),box=box.tolist(),suspicious=bool(suspicious)))
                if suspicious and not any(s[0]==row['path'] for s in samples):
                    view=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(view)
                    d.polygon([tuple(p) for p in box],outline='red',width=2);samples.append((row['path'],txt,view))
        record=dict(path=row['path'],source_id=source,split=row['split'],sampled_frames=sampled,
                    suspicious=any(d['suspicious'] for d in detections),detections=detections,training_approved=False)
        results.append(record)
        with (out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        if (i+1)%20==0:print(json.dumps(dict(clips=i+1,flagged=sum(r['suspicious'] for r in results))),flush=True)
    if cap is not None:cap.release()
    report=dict(warning='OCR triage, not complete caption detection. Names/HUD can trigger flags; misses remain possible.',
                arguments=vars(args),clips=len(results),flagged=sum(r['suspicious'] for r in results),seconds=time.monotonic()-started,rows=results)
    (out/'caption-screen.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for page in range((len(samples)+19)//20):
        subset=samples[page*20:(page+1)*20];sheet=Image.new('RGB',(1000,240*((len(subset)+3)//4)),'white');d=ImageDraw.Draw(sheet)
        for j,(path,txt,im) in enumerate(subset):
            x=j%4*250;y=j//4*240;d.text((x+3,y+3),Path(path).name[:36],fill='black');d.text((x+3,y+18),txt[:30],fill='black')
            im.thumbnail((200,200));sheet.paste(im,(x+3,y+36))
        sheet.save(out/f'flagged-{page+1:02}.png')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('out');p.add_argument('--raw',default='data/raw')
    p.add_argument('--every-frame',action='store_true');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--padding',type=int,default=24,help='Read outside crop to recognize partial captions; only intersecting text counts')
    p.add_argument('--previous-screen',help='Rescreen only negatives from a complete earlier screening pass')
    run(p.parse_args())
