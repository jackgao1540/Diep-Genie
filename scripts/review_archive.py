"""Compact, immutable archive overview for class/source triage, never approval."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from genie_diep.data import file_hash


def run(args):
    root=Path(args.raw);out=Path(args.out)
    if out.exists():raise FileExistsError('Choose a new overview directory')
    rows=[r for r in json.loads(Path(args.catalog).read_text(encoding='utf-8'))
          if r['selection']!='excluded_other_game']
    out.mkdir(parents=True);index=[]
    for page in range((len(rows)+3)//4):
        sheet=Image.new('RGB',(1280,4*200),'white');draw=ImageDraw.Draw(sheet)
        for slot,row in enumerate(rows[page*4:page*4+4]):
            paths=[p for ext in ('mp4','webm','mkv') if (p:=root/f"{row['id']}.{ext}").exists()]
            if not paths:raise FileNotFoundError(row['id'])
            cap=cv2.VideoCapture(str(paths[0]));fps=cap.get(cv2.CAP_PROP_FPS)
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps<=0:raise ValueError('Invalid FPS')
            duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
            times=np.linspace(5,max(5,duration-10),10);y=slot*200
            draw.text((4,y+2),f"#{page*4+slot} {row['id']} {duration/60:.1f}min {row['split']} | {row['title'][:120]}",fill='black')
            # One full-frame identity check and ten player closeups per source.
            for col,t in enumerate(times):
                cap.set(cv2.CAP_PROP_POS_MSEC,float(t*1000));ok,frame=cap.read()
                if not ok:raise RuntimeError(f"Cannot sample {row['id']} at {t}")
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                side=round(height*72/360);cx=width//2;cy=height//2
                close=Image.fromarray(rgb[cy-side//2:cy+side//2,cx-side//2:cx+side//2])
                sheet.paste(close.resize((96,96)),(col*100,y+44))
                draw.text((col*100+3,y+28),f'{t:.1f}s',fill='black')
                if col==2:
                    full=Image.fromarray(rgb);full.thumbnail((256,144))
                    sheet.paste(full,(1020,y+42))
            cap.release()
            index.append({**row,'page':page+1,'slot':slot,'overview_index':page*4+slot,
                'width':width,'height':height,'duration':duration,'sample_times':times.tolist(),
                'video_sha256':file_hash(paths[0]),'media_path':paths[0].as_posix(),
                'warning':'Sparse class/source triage only; does not approve intervals or infer action labels.'})
        sheet.save(out/f'overview-{page+1:02}.png')
        print(json.dumps(dict(page=page+1,sources=len(index))),flush=True)
    (out/'index.json').write_text(json.dumps(dict(catalog_sha256=file_hash(args.catalog),sources=index),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('catalog');p.add_argument('out');p.add_argument('--raw',default='data/raw')
    run(p.parse_args())
