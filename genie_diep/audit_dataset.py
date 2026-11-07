"""Local data-coverage counts and deterministic multi-frame curation sheets.

Color groups are heuristic diagnostics, not labels supplied to the video model.
Counts include repeated appearances of the same object across frames.
"""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .object_quality import components


def audit(root,out,page_size=20):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for p in sorted(root.glob('*.jsonl'))
          for line in p.read_text(encoding='utf-8').splitlines() if line]
    rows.sort(key=lambda r:(r['source_id'],r['start']))
    groups=defaultdict(lambda:dict(clips=0,frames=0,objects=Counter(),pixels=Counter(),frames_with=Counter()))
    details=[]
    for page in range(math.ceil(len(rows)/page_size)):
        subset=rows[page*page_size:(page+1)*page_size]
        sheet=Image.new('RGB',(960,110*math.ceil(len(subset)/4)),'white');draw=ImageDraw.Draw(sheet)
        for n,row in enumerate(subset):
            path=(root/row['path']).resolve()
            if not path.is_relative_to(root.resolve()):raise ValueError('Clip path escapes root')
            clip=np.load(path,allow_pickle=False)
            key=f"{row['split']}/{row['source_id']}";group=groups[key];group['clips']+=1
            clip_counts=Counter()
            for frame in clip:
                items=components(frame);group['frames']+=1
                present=set()
                for item in items:
                    color=item['color'];group['objects'][color]+=1;group['pixels'][color]+=item['area']
                    clip_counts[color]+=1;present.add(color)
                group['frames_with'].update(present)
            details.append(dict(path=row['path'],source_id=row['source_id'],split=row['split'],start=row['start'],end=row['end'],objects=dict(clip_counts),reviewed=False))
            xx=(n%4)*240;yy=(n//4)*110
            draw.text((xx+2,yy+2),f"{row['source_id']} {row['start']:.1f}-{row['end']:.1f}s",fill='black')
            draw.text((xx+2,yy+16),row['split'],fill='black')
            for j,t in enumerate((0,len(clip)//2,len(clip)-1)):
                sheet.paste(Image.fromarray(clip[t]).resize((78,78),Image.Resampling.NEAREST),(xx+j*80,yy+30))
        sheet.save(out/f'curation-{page+1:02}.png')
    report=dict(root=str(root),warning='Heuristic color components; appearances across frames are not independent objects. Sheets do not constitute automatic curation.',groups=dict(groups),clips=details)
    (out/'coverage.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['groups'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('out')
    audit(**vars(parser.parse_args()))
