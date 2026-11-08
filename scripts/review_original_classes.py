"""Review tank classes from original frames rather than reduced proxies."""
import argparse
import json
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
from genie_diep.data import file_hash
from genie_diep.class_hud import class_bar_crop
from genie_diep.curation_reviews import load_class_review
from scripts.review_longform import candidates


def unreviewed_rows(rows, reviews):
    decisions={}
    for path in reviews:
        for r in load_class_review(path):
            key=(r['source_id'],r['block_id'])
            if key in decisions:raise ValueError('Repeated prior class decision')
            decisions[key]=r
    remaining=[]
    for r in rows:
        previous=decisions.get((r['source_id'],r['block_id']))
        if previous:
            if previous['proxy_sha256']!=r['proxy_sha256']:raise ValueError('Prior class evidence changed')
        else:remaining.append(r)
    return remaining


def run(args):
    cv2.setNumThreads(1)
    out=Path(args.out)
    if out.exists():raise FileExistsError('Choose a new immutable review directory')
    sources={s['id']:s for p in args.recipes for s in json.loads(Path(p).read_text(encoding='utf-8'))['sources']}
    rows=candidates(args.screens,new_only=args.new_only)
    if args.source:rows=[r for r in rows if r['source_id'] in args.source]
    rows=unreviewed_rows(rows,args.exclude_reviews)
    if args.limit:rows=rows[:args.limit]
    caps={}
    for sid in {r['source_id'] for r in rows}:
        source=sources[sid]
        if file_hash(source['media_path'])!=source['video_sha256']:raise ValueError('Original media changed')
        caps[sid]=cv2.VideoCapture(source['media_path'])
    out.mkdir(parents=True)
    for page in range((len(rows)+15)//16):
        sheet=Image.new('RGB',(1344,1344),'white');draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(rows[page*16:(page+1)*16]):
            x=j%2*672;y=j//2*168;sid=row['source_id'];source=sources[sid]
            if row['video_sha256']!=source['video_sha256'] or row['crop']!=source['crop']:raise ValueError('Source mismatch')
            if row['fps']!=10 or len(row['source_frame_indices'])!=40:raise ValueError('Expected four-second 10FPS screening blocks')
            draw.text((x+2,y+1),f"#{page*16+j} {sid} {row['start']:.1f}s",fill='black')
            cap=caps[sid];a,b,w,h=row['crop']
            for k,fi in enumerate((0,10,20,30)):
                cap.set(cv2.CAP_PROP_POS_FRAMES,row['source_frame_indices'][fi]);ok,frame=cap.read()
                if not ok:raise RuntimeError('Original frame decode failed')
                im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));side=round(min(w,h)*.48)
                cx=a+w//2;cy=b+h//2
                sheet.paste(im.crop((cx-side//2,cy-side//2,cx+side//2,cy+side//2)).resize((128,128)),(x+k*130,y+18))
                if k==2:
                    sheet.paste(im.crop((a,b,a+w,b+h)).resize((128,128)),(x+528,y+18))
                    bar=class_bar_crop(frame)
                    if bar is not None:sheet.paste(Image.fromarray(cv2.cvtColor(bar,cv2.COLOR_BGR2RGB)).resize((256,20)),(x+270,y+147))
            row.update(review_index=page*16+j,page=page+1,slot=j)
        sheet.save(out/f'class-original-{page+1:03}.png')
        print(json.dumps(dict(page=page+1,total_pages=(len(rows)+15)//16)),flush=True)
    for cap in caps.values():cap.release()
    (out/'index.json').write_text(json.dumps(dict(arguments=vars(args),rows=rows,
        warning='Four original-frame player crops and context per block. Native class-bar crop is visual assistance, not OCR proof. Full-frame sample audit remains required.'),indent=2),encoding='utf-8')
    print(json.dumps(dict(clips=len(rows),pages=(len(rows)+15)//16)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('screens',nargs='+');p.add_argument('--out',required=True)
    p.add_argument('--recipes',nargs='+',required=True);p.add_argument('--source',nargs='+');p.add_argument('--limit',type=int)
    p.add_argument('--new-only',action='store_true')
    p.add_argument('--exclude-reviews',nargs='*',default=[],help='Skip unchanged blocks with accepted or rejected class decisions')
    run(p.parse_args())
