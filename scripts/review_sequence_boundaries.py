"""Content-bound visual review of every proposed join in a sequence report."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from genie_diep.data import file_hash


def run(args):
    out=Path(args.out)
    if out.exists():raise FileExistsError('Choose a new boundary review directory')
    data=json.loads(Path(args.index).read_text(encoding='utf-8'))
    rows={(r['source_id'],r['block_id']):r for r in data['rows']}
    sequences=json.loads(Path(args.sequences).read_text(encoding='utf-8'))['runs'];pairs=[]
    for seq in sequences:
        for a,b in zip(seq['blocks'],seq['blocks'][1:]):
            first=rows[(seq['source_id'],a)];second=rows[(seq['source_id'],b)]
            pairs.append(dict(source_id=seq['source_id'],before=a,after=b,time=second['start'],
                before_path=first['proxy_path'],after_path=second['proxy_path'],
                before_sha256=first['proxy_sha256'],after_sha256=second['proxy_sha256']))
    out.mkdir(parents=True)
    for page in range((len(pairs)+95)//96):
        image=Image.new('RGB',(1280,1152),'white');draw=ImageDraw.Draw(image)
        for j,pair in enumerate(pairs[page*96:(page+1)*96]):
            x=j%8*160;y=j//8*96;draw.text((x,y),f"#{page*96+j} {pair['time']:.1f}s",fill='black')
            for side,index,offset in [('before',-1,0),('after',0,80)]:
                path=pair[side+'_path']
                if file_hash(path)!=pair[side+'_sha256']:raise ValueError('Boundary proxy changed')
                with np.load(path,allow_pickle=False) as z:frame=z['frames'][index]
                image.paste(Image.fromarray(frame).resize((80,80)),(x+offset,y+16))
        image.save(out/f'boundary-{page+1:03}.png')
    (out/'index.json').write_text(json.dumps(dict(arguments=vars(args),
        source_index_sha256=file_hash(args.index),sequences_sha256=file_hash(args.sequences),pairs=pairs),indent=2),encoding='utf-8')
    print(json.dumps(dict(boundaries=len(pairs),pages=(len(pairs)+95)//96)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('index');p.add_argument('sequences');p.add_argument('out');run(p.parse_args())
