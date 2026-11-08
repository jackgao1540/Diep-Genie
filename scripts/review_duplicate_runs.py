"""Render original start/middle/end crops for flagged near-duplicate runs."""
import argparse
import json
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
from genie_diep.data import file_hash


def run(args):
    out=Path(args.out)
    if out.exists():raise FileExistsError('Choose an immutable review directory')
    report=json.loads(Path(args.report).read_text(encoding='utf-8'))
    sources={s['id']:s for path in args.recipes for s in json.loads(Path(path).read_text(encoding='utf-8'))['sources']}
    flags=report['near_runs'];used={r['source_'+side] for r in flags for side in ('a','b')}
    for source in used:
        s=sources[source]
        if file_hash(s['media_path'])!=s['video_sha256']:raise ValueError('Original media changed')
    cv2.setNumThreads(1);out.mkdir(parents=True)
    for page in range((len(flags)+7)//8):
        sheet=Image.new('RGB',(800,1280),'white');draw=ImageDraw.Draw(sheet)
        for slot,r in enumerate(flags[page*8:(page+1)*8]):
            x=slot%2*400;y=slot//2*320
            for k,side in enumerate(('a','b')):
                s=sources[r['source_'+side]];start=r['start_'+side];end=r['end_'+side]
                draw.text((x,y+k*155),f'#{page*8+slot} {side} {s["id"]} {start:.2f}..{end:.2f}',fill='black')
                cap=cv2.VideoCapture(s['media_path'])
                for j,t in enumerate((start,(start+end)/2,end)):
                    cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read()
                    if not ok:raise RuntimeError('Cannot decode original')
                    cx,cy,w,h=s['crop'];crop=cv2.cvtColor(frame[cy:cy+h,cx:cx+w],cv2.COLOR_BGR2RGB)
                    sheet.paste(Image.fromarray(crop).resize((128,128)),(x+j*128,y+k*155+20))
                cap.release()
        sheet.save(out/f'duplicates-{page+1:03}.png')
    (out/'index.json').write_text(json.dumps(dict(report=args.report,report_sha256=file_hash(args.report),rows=flags,
        method='Original start/middle/end crops; sparse comparison, not automatic duplicate decisions.'),indent=2),encoding='utf-8')
    print(json.dumps(dict(flags=len(flags),pages=(len(flags)+7)//8)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('report');p.add_argument('out');p.add_argument('--recipes',nargs='+',required=True)
    run(p.parse_args())
