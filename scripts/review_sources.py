"""Deterministic full-frame and center-crop sheets for local source triage.

No source is approved automatically. These samples locate candidate intervals;
the exact selected clips still need caption/cut/class review before training.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw


def run(args):
    root=Path(args.raw);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for video_id in args.ids:
        files=[p for ext in ('mp4','webm','mkv') if (p:=root/f'{video_id}.{ext}').exists()]
        if not files:raise FileNotFoundError(video_id)
        meta=json.loads((root/f'{video_id}.info.json').read_text(encoding='utf-8'))
        cap=cv2.VideoCapture(str(files[0]));fps=cap.get(cv2.CAP_PROP_FPS)
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
        side=round(min(width,height)*200/360);crop=[(width-side)//2,(height-side)//2,side,side]
        times=np.linspace(2,max(2,duration-4),args.samples)
        canvas=Image.new('RGB',(1200,220*((len(times)+2)//3)+32),'white');draw=ImageDraw.Draw(canvas)
        draw.text((4,4),f"{video_id} | {meta.get('title')} | center crop {crop}",fill='black')
        for i,t in enumerate(times):
            cap.set(cv2.CAP_PROP_POS_MSEC,float(t*1000));ok,frame=cap.read()
            if not ok:continue
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);im=Image.fromarray(rgb)
            im.thumbnail((256,150));xx=i%3*400;yy=i//3*220+32
            draw.text((xx+4,yy+4),f'{t:.1f}s',fill='black');canvas.paste(im,(xx+2,yy+24))
            x,y,w,h=crop;small=Image.fromarray(rgb[y:y+h,x:x+w]).resize((128,128))
            canvas.paste(small,(xx+268,yy+24));draw.text((xx+268,yy+158),'candidate crop',fill='black')
        cap.release();canvas.save(out/f'{video_id}.png')
        (out/f'{video_id}.json').write_text(json.dumps(dict(id=video_id,title=meta.get('title'),
             width=width,height=height,duration=duration,sample_times=times.tolist(),suggested_crop=crop,
             warning='Automatic center crop proposal only; review facecams, HUD, captions, scale and tank class.'),indent=2),encoding='utf-8')
        print(video_id,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('ids',nargs='+');p.add_argument('--raw',default='data/raw')
    p.add_argument('--out',default='runs/source-review');p.add_argument('--samples',type=int,default=15)
    run(p.parse_args())
