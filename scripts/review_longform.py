"""Immutable candidate review sheets with content-bound indexes, not approval."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image,ImageDraw
from genie_diep.data import file_hash
from genie_diep.curation_reviews import load_class_review


def candidates(screens,class_reviews=(),new_only=False):
    decisions={}
    for path in class_reviews:
        for r in load_class_review(path):
            key=(r['source_id'],r['block_id'])
            if key in decisions:raise ValueError('Repeated class decision')
            decisions[key]=r
    rows=[];seen=set()
    for root in screens:
        for path in sorted(Path(root).glob('*/blocks.jsonl')):
            for line in path.read_text(encoding='utf-8').splitlines():
                r=json.loads(line)
                if r['reasons']:continue
                if new_only and r.get('reused_strict_screen'):continue
                key=(r['source_id'],r['block_id'])
                if key in seen:raise ValueError('Repeated source block')
                seen.add(key)
                if class_reviews:
                    decision=decisions.get(key)
                    if not decision or not decision['class_candidate']:continue
                    if decision['proxy_sha256']!=r['proxy_sha256']:raise ValueError('Class review hash mismatch')
                r['proxy_path']=(path.parent/r['proxy']).as_posix()
                if file_hash(r['proxy_path'])!=r['proxy_sha256']:raise ValueError('Changed candidate')
                rows.append(r)
    return rows


def run(args):
    out=Path(args.out)
    if out.exists():raise FileExistsError('Choose a new immutable review directory')
    rows=candidates([args.screen]+args.extra_screens,args.class_reviews,args.new_only)
    if args.source:rows=[r for r in rows if r['source_id'] in args.source]
    if args.limit and len(rows)>args.limit:
        # Source-stratified round robin random sample, reproducible and disclosed.
        groups={};rng=random.Random(args.seed)
        for r in rows:groups.setdefault(r['source_id'],[]).append(r)
        for g in groups.values():rng.shuffle(g)
        sample=[]
        while groups and len(sample)<args.limit:
            for source in list(groups):
                if len(sample)>=args.limit:break
                sample.append(groups[source].pop())
                if not groups[source]:del groups[source]
        rows=sample
    out.mkdir(parents=True);per_page=8 if args.mode=='audit' else 64
    for page in range((len(rows)+per_page-1)//per_page):
        sheet=Image.new('RGB',(1280,1152 if args.mode=='audit' else 1280),'white');draw=ImageDraw.Draw(sheet)
        for slot,row in enumerate(rows[page*per_page:(page+1)*per_page]):
            frames=np.load(row['proxy_path'],allow_pickle=False)['frames']
            if args.mode=='audit':
                x=slot%2*640;y=slot//2*288
                draw.text((x+2,y+2),f"#{page*per_page+slot} {row['source_id']} {row['start']:.2f}s",fill='black')
                for i,frame in enumerate(frames):
                    sheet.paste(Image.fromarray(frame).resize((64,64)),(x+i%10*64,y+26+i//10*64))
            else:
                x=slot%4*320;y=slot//4*80
                draw.text((x+2,y+1),f"#{page*per_page+slot} {row['source_id']} {row['start']:.1f}s",fill='black')
                for i,index in enumerate((0,10,20,30)):
                    frame=frames[min(index,len(frames)-1)];h,w=frame.shape[:2];side=round(h*.44)
                    close=frame[(h-side)//2:(h+side)//2,(w-side)//2:(w+side)//2]
                    sheet.paste(Image.fromarray(close).resize((60,60)),(x+i*62,y+19))
                sheet.paste(Image.fromarray(frames[len(frames)//2]).resize((60,60)),(x+250,y+19))
            row.update(review_index=page*per_page+slot,page=page+1,slot=slot)
        sheet.save(out/f'{args.mode}-{page+1:03}.png')
    (out/'index.json').write_text(json.dumps(dict(arguments=vars(args),rows=rows,
        warning='Class sheets are sparse triage. Audit sheets show all40 retained frames at reduced size. Original source retained for close inspection.'),indent=2),encoding='utf-8')
    print(json.dumps(dict(clips=len(rows),pages=(len(rows)+per_page-1)//per_page,mode=args.mode)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('screen');p.add_argument('out')
    p.add_argument('--mode',choices=['class','audit'],default='audit');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--source',nargs='+');p.add_argument('--seed',type=int,default=20261107)
    p.add_argument('--new-only',action='store_true',help='Review recovered candidates, excluding unchanged strict negatives')
    p.add_argument('--extra-screens',nargs='*',default=[])
    p.add_argument('--class-reviews',nargs='*',default=[],help='Only include hash-matched class candidates')
    run(p.parse_args())
