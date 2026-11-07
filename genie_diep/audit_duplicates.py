"""Exact and conservative near-duplicate triage across dataset splits.

Near matches need visual review: sparse game scenes can look alike. This screen
does not prove absence of edited, offset, rescaled or speed-changed duplicates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw


def signature(frames):
    bits=[]
    for frame in frames:
        gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        dct=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8].flatten()
        bits.append(dct>np.median(dct[1:]))
    return np.stack(bits)


def run(args):
    root=Path(args.data);rows=[json.loads(line) for p in sorted(root.glob('*.jsonl'))
        for line in p.read_text(encoding='utf-8').splitlines() if line]
    clips=[];hashes=[];features=[]
    for row in rows:
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('Unsafe clip path')
        clip=np.load(path,allow_pickle=False);hashes.append(hashlib.sha256(clip.tobytes()).hexdigest())
        frames=clip[[0,len(clip)//2,len(clip)-1]];clips.append(frames);features.append(signature(frames))
    features=np.stack(features);matches=[]
    for i,a in enumerate(rows):
        distances=np.not_equal(features,features[i]).sum(-1).mean(-1)
        for j in np.flatnonzero(distances<=6):
            if j<=i or rows[j]['split']==a['split']:continue
            mse=float(np.square((clips[i].astype(np.float32)-clips[j].astype(np.float32))/255).mean())
            exact=hashes[i]==hashes[j]
            if exact or mse<=.001:
                matches.append(dict(first=a['path'],second=rows[j]['path'],first_index=i,second_index=int(j),
                                    exact_pixels=exact,mean_hash_distance=float(distances[j]),sample_mse=mse))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    report=dict(clips=len(rows),exact_cross_split=sum(m['exact_pixels'] for m in matches),
                near_candidates=len(matches),warning=__doc__,matches=matches)
    (out/'duplicates.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for page in range((len(matches)+7)//8):
        subset=matches[page*8:(page+1)*8];sheet=Image.new('RGB',(800,156*len(subset)),'white');d=ImageDraw.Draw(sheet)
        for n,m in enumerate(subset):
            yy=n*156;d.text((3,yy+2),f"{m['first']} | {m['second']} | mse {m['sample_mse']:.6f}",fill='black')
            for side,key in enumerate(('first_index','second_index')):
                for t,frame in enumerate(clips[m[key]]):sheet.paste(Image.fromarray(frame).resize((128,128)),(side*400+t*130,yy+23))
        sheet.save(out/f'matches-{page+1:02}.png')
    print(json.dumps({k:v for k,v in report.items() if k not in ('matches','warning')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('out');run(p.parse_args())
