"""Freeze a visual review batch of OCR-negative candidates; show every frame.

The generated index binds each displayed clip to a content hash. Nothing is
approved automatically. Explicit decisions can later construct a curated set.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def run(args):
    root=Path(args.data);out=Path(args.out)
    if out.exists():raise FileExistsError('Review batches are immutable; choose another directory')
    rows=[json.loads(line) for p in sorted(root.glob('*.jsonl')) for line in p.read_text(encoding='utf-8').splitlines() if line]
    screening=Path(args.screen)
    if screening.suffix=='.jsonl':screen=[json.loads(x) for x in screening.read_text(encoding='utf-8').splitlines() if x]
    else:screen=json.loads(screening.read_text(encoding='utf-8'))['rows']
    screened={r['path']:r for r in screen};excluded=set()
    for batch in args.exclude_batches:
        excluded.update(r['path'] for r in json.loads(Path(batch).read_text())['clips'])
    selected=[r for r in rows if r['path'] in screened and not screened[r['path']]['suspicious']
              and screened[r['path']]['sampled_frames']==r['shape'][0] and r['path'] not in excluded]
    if args.limit:selected=selected[:args.limit]
    out.mkdir(parents=True);index=[]
    for page in range((len(selected)+11)//12):
        subset=selected[page*12:(page+1)*12]
        sheet=Image.new('RGB',(816,290*((len(subset)+2)//3)),'white');d=ImageDraw.Draw(sheet)
        for cell,row in enumerate(subset):
            path=(root/row['path']).resolve()
            if not path.is_relative_to(root.resolve()):raise ValueError('Unsafe candidate path')
            data=path.read_bytes();clip=np.load(path,allow_pickle=False)
            if len(clip)!=16:raise ValueError('Review layout expects 16-frame clips')
            number=page*12+cell;xx=cell%3*272;yy=cell//3*290
            d.text((xx+2,yy+2),f"#{number} {row['source_id']} {row['start']:.2f}s",fill='black')
            d.text((xx+2,yy+16),row['split']+'; all 16 frames, left to right',fill='black')
            for t,frame in enumerate(clip):
                sheet.paste(Image.fromarray(frame).resize((64,64)),(xx+2+t%4*66,yy+32+t//4*64))
            index.append(dict(index=number,page=page+1,cell=cell,path=row['path'],sha256=hashlib.sha256(data).hexdigest(),
                              source_id=row['source_id'],split=row['split'],start=row['start'],end=row['end']))
        sheet.save(out/f'review-{page+1:02}.png')
    (out/'index.json').write_text(json.dumps(dict(data=str(root),screen=str(screening),
       warning='OCR-negative is not clean approval. Review every displayed frame for wrong classes, occlusion, edits and overlays.',clips=index),indent=2),encoding='utf-8')
    print(json.dumps(dict(review_clips=len(index),pages=(len(index)+11)//12)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('screen');p.add_argument('out')
    p.add_argument('--exclude-batches',nargs='*',default=[]);p.add_argument('--limit',type=int,default=0)
    run(p.parse_args())
